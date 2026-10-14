"""SQLite persistence: runs, hypotheses, experiments, code versions, executions, LLM calls.

Every write commits immediately so an interrupted run loses at most the
experiments that were in flight; those are marked `interrupted` on resume and
their hypotheses re-queued.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE, domain TEXT, objective TEXT,
    config_json TEXT, limits_json TEXT, system_revision TEXT, integrity_json TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY, run_id INTEGER, item TEXT, old_value TEXT, new_value TEXT,
    approved_by TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY, run_id INTEGER, title TEXT, rationale TEXT, method TEXT,
    expected_metric REAL, priority REAL, parent_experiment_ids TEXT,
    status TEXT DEFAULT 'pending',      -- pending | claimed | done
    llm_call_id INTEGER, created_at REAL
);
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY, run_id INTEGER, hypothesis_id INTEGER, is_baseline INTEGER DEFAULT 0,
    status TEXT,                        -- running | completed | failed | interrupted
    seed INTEGER, system_revision TEXT, attempts INTEGER DEFAULT 0, code_version_id INTEGER,
    val_score REAL, test_score REAL, baseline_val REAL, improvement REAL, success INTEGER,
    verdict TEXT, analysis TEXT, lessons TEXT, error TEXT,
    started_at REAL, finished_at REAL
);
CREATE TABLE IF NOT EXISTS code_versions (
    id INTEGER PRIMARY KEY, experiment_id INTEGER, attempt INTEGER, parent_version_id INTEGER,
    code TEXT, code_sha256 TEXT, explanation TEXT, llm_call_id INTEGER, created_at REAL
);
CREATE TABLE IF NOT EXISTS executions (
    id INTEGER PRIMARY KEY, experiment_id INTEGER, code_version_id INTEGER, attempt INTEGER,
    image TEXT, image_id TEXT, command TEXT, exit_code INTEGER, timed_out INTEGER,
    duration_s REAL, stdout_tail TEXT, stderr_tail TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY, run_id INTEGER, experiment_id INTEGER, role TEXT, provider TEXT,
    model TEXT, system TEXT, prompt TEXT, response TEXT, input_tokens INTEGER,
    output_tokens INTEGER, cost_usd REAL, error TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS agent_steps (
    id INTEGER PRIMARY KEY, run_id INTEGER, experiment_id INTEGER, hypothesis_id INTEGER,
    phase TEXT,                         -- session | propose | implement | execute | score | reflect | decide | stop
    attempt INTEGER, worker INTEGER,
    status TEXT,                        -- running | ok | failed | skipped
    summary TEXT, detail_json TEXT, started_at REAL, finished_at REAL
);
CREATE TABLE IF NOT EXISTS permission_requests (
    id INTEGER PRIMARY KEY, run_id INTEGER, experiment_id INTEGER, request TEXT,
    status TEXT DEFAULT 'pending', created_at REAL
);
"""


class Memory:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def _insert(self, table: str, **cols) -> int:
        cols.setdefault("created_at", time.time())
        keys = ", ".join(cols)
        qs = ", ".join("?" for _ in cols)
        return self.db.execute(f"INSERT INTO {table} ({keys}) VALUES ({qs})", list(cols.values())).lastrowid

    def _update(self, table: str, row_id: int, **cols) -> None:
        sets = ", ".join(f"{k} = ?" for k in cols)
        self.db.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*cols.values(), row_id])

    def rows(self, sql: str, *args) -> list[dict]:
        return [dict(r) for r in self.db.execute(sql, args)]

    def one(self, sql: str, *args) -> dict | None:
        r = self.db.execute(sql, args).fetchone()
        return dict(r) if r else None

    # --- runs -------------------------------------------------------------
    def get_run(self, name: str) -> dict | None:
        return self.one("SELECT * FROM runs WHERE name = ?", name)

    def create_run(self, name, domain, objective, config, limits, revision, integrity) -> int:
        return self._insert("runs", name=name, domain=domain, objective=objective,
                            config_json=json.dumps(config), limits_json=json.dumps(limits),
                            system_revision=revision, integrity_json=json.dumps(integrity))

    def update_run_limits(self, run_id: int, config: dict, limits: dict) -> None:
        self._update("runs", run_id, config_json=json.dumps(config), limits_json=json.dumps(limits))

    def record_approval(self, run_id, item, old, new, who) -> None:
        self._insert("approvals", run_id=run_id, item=item, old_value=str(old), new_value=str(new),
                     approved_by=who)

    def recover_interrupted(self, run_id: int) -> int:
        """Mark in-flight experiments interrupted and re-queue their hypotheses."""
        stuck = self.rows("SELECT id, hypothesis_id FROM experiments WHERE run_id=? AND status='running'", run_id)
        for e in stuck:
            self._update("experiments", e["id"], status="interrupted", finished_at=time.time())
            if e["hypothesis_id"]:
                self._update("hypotheses", e["hypothesis_id"], status="pending")
        self.db.execute("UPDATE hypotheses SET status='pending' WHERE run_id=? AND status='claimed'", (run_id,))
        self.db.execute("UPDATE agent_steps SET status='interrupted', finished_at=? WHERE run_id=? AND status='running'",
                        (time.time(), run_id))
        return len(stuck)

    # --- hypotheses -------------------------------------------------------
    def add_hypothesis(self, run_id: int, h: dict, llm_call_id: int | None) -> int:
        return self._insert("hypotheses", run_id=run_id, title=h["title"], rationale=h["rationale"],
                            method=h["method"], expected_metric=h["expected_metric"],
                            priority=h["priority"], parent_experiment_ids=json.dumps(h["parent_experiment_ids"]),
                            llm_call_id=llm_call_id)

    def claim_next_hypothesis(self, run_id: int) -> dict | None:
        h = self.one("SELECT * FROM hypotheses WHERE run_id=? AND status='pending' "
                     "ORDER BY priority DESC, id ASC LIMIT 1", run_id)
        if h:
            self._update("hypotheses", h["id"], status="claimed")
        return h

    def set_hypothesis_status(self, hid: int, status: str) -> None:
        self._update("hypotheses", hid, status=status)

    def hypotheses(self, run_id: int) -> list[dict]:
        return self.rows("SELECT * FROM hypotheses WHERE run_id=? ORDER BY id", run_id)

    # --- experiments ------------------------------------------------------
    def start_experiment(self, run_id, hypothesis_id, seed, revision, is_baseline=False) -> int:
        return self.db.execute(
            "INSERT INTO experiments (run_id, hypothesis_id, seed, system_revision, is_baseline, status, started_at)"
            " VALUES (?,?,?,?,?, 'running', ?)",
            (run_id, hypothesis_id, seed, revision, int(is_baseline), time.time())).lastrowid

    def finish_experiment(self, exp_id: int, **cols) -> None:
        self._update("experiments", exp_id, finished_at=time.time(), **cols)

    def update_experiment(self, exp_id: int, **cols) -> None:
        self._update("experiments", exp_id, **cols)

    def experiments(self, run_id: int) -> list[dict]:
        return self.rows(
            "SELECT e.*, h.title, h.method, h.rationale, h.expected_metric, h.parent_experiment_ids "
            "FROM experiments e LEFT JOIN hypotheses h ON h.id = e.hypothesis_id "
            "WHERE e.run_id=? ORDER BY e.id", run_id)

    def baseline(self, run_id: int) -> dict | None:
        return self.one("SELECT * FROM experiments WHERE run_id=? AND is_baseline=1 AND status='completed'", run_id)

    def count_experiments(self, run_id: int) -> int:
        """Non-baseline experiments that count against max_experiments (finished or in flight)."""
        return self.one("SELECT COUNT(*) n FROM experiments WHERE run_id=? AND is_baseline=0 "
                        "AND status IN ('running','completed','failed')", run_id)["n"]

    # --- code / executions / llm -----------------------------------------
    def add_code_version(self, **cols) -> int:
        return self._insert("code_versions", **cols)

    def add_execution(self, **cols) -> int:
        return self._insert("executions", **cols)

    def add_llm_call(self, **cols) -> int:
        return self._insert("llm_calls", **cols)

    def total_cost(self, run_id: int) -> float:
        return self.one("SELECT COALESCE(SUM(cost_usd), 0) c FROM llm_calls WHERE run_id=?", run_id)["c"]

    # --- agent step log -------------------------------------------------------
    def start_step(self, run_id: int, phase: str, *, experiment_id: int | None = None,
                   hypothesis_id: int | None = None, attempt: int | None = None, worker: int | None = None,
                   summary: str = "") -> int:
        return self.db.execute(
            "INSERT INTO agent_steps (run_id, experiment_id, hypothesis_id, phase, attempt, worker, status, summary,"
            " detail_json, started_at) VALUES (?,?,?,?,?,?, 'running', ?, '{}', ?)",
            (run_id, experiment_id, hypothesis_id, phase, attempt, worker, summary, time.time())).lastrowid

    def finish_step(self, step_id: int, status: str, summary: str | None = None, **detail) -> None:
        row = self.one("SELECT detail_json, summary FROM agent_steps WHERE id=?", step_id)
        merged = {**json.loads(row["detail_json"] or "{}"), **detail}
        self._update("agent_steps", step_id, status=status, finished_at=time.time(),
                     summary=row["summary"] if summary is None else summary,
                     detail_json=json.dumps(merged, default=str))

    def steps(self, run_id: int) -> list[dict]:
        return self.rows("SELECT * FROM agent_steps WHERE run_id=? ORDER BY started_at, id", run_id)

    def add_permission_request(self, run_id, experiment_id, request) -> int:
        return self._insert("permission_requests", run_id=run_id, experiment_id=experiment_id, request=request)
