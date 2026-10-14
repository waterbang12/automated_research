"""Agent trace and run checks: what the viewer's Run page shows.

A run is presented as one agent working through steps:
  session -> propose -> (implement -> execute -> score)+ -> reflect -> decide -> ... -> stop

Runs recorded since `agent_steps` existed have a live, exact trace written by the controller.
Older runs get a trace reconstructed from llm_calls / executions / experiments, marked
`reconstructed` (end times only, no in-flight steps). Everything here is read-only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .memory import Memory

PHASES = ("session", "propose", "implement", "execute", "score", "reflect", "decide", "stop")
ROLE_PHASE = {"researcher": "propose", "coder": "implement", "analyst": "reflect"}


def timeline(m: Memory, run: dict) -> dict:
    rid = run["id"]
    titles = {h["id"]: h["title"] for h in m.hypotheses(rid)}
    exp_h = {e["id"]: e["hypothesis_id"] for e in m.rows("SELECT id, hypothesis_id FROM experiments WHERE run_id=?", rid)}
    recorded = _has_steps(m) and m.steps(rid)
    steps = [_recorded(s) for s in recorded] if recorded else reconstruct(m, rid)
    for s in steps:
        hid = s.get("hypothesis_id") or exp_h.get(s.get("experiment_id"))
        s["hypothesis_id"] = hid
        s["hypothesis_title"] = titles.get(hid)
    return {"source": "recorded" if recorded else "reconstructed", "steps": steps, "state": state(m, run, steps)}


def _has_steps(m: Memory) -> bool:
    return bool(m.one("SELECT name FROM sqlite_master WHERE type='table' AND name='agent_steps'"))


def _recorded(s: dict) -> dict:
    return {**s, "detail": json.loads(s.pop("detail_json") or "{}"), "origin": "recorded"}


def reconstruct(m: Memory, rid: int) -> list[dict]:
    """Best-effort trace for runs recorded before agent_steps existed."""
    out = []
    proposed: dict[int, list[str]] = {}
    for h in m.hypotheses(rid):
        proposed.setdefault(h["llm_call_id"], []).append(h["title"])
    for c in m.rows("SELECT id, role, experiment_id, error, created_at FROM llm_calls WHERE run_id=? ORDER BY id", rid):
        summary = (c["error"] or "")[:200]
        if not summary and c["role"] == "researcher" and proposed.get(c["id"]):
            summary = f"proposed {len(proposed[c['id']])}: " + "; ".join(proposed[c["id"]])
        out.append({"id": f"llm{c['id']}", "phase": ROLE_PHASE.get(c["role"], c["role"]),
                    "experiment_id": c["experiment_id"], "attempt": None, "worker": None,
                    "status": "failed" if c["error"] else "ok", "summary": summary,
                    "detail": {"llm_call_ids": [c["id"]]}, "started_at": None, "finished_at": c["created_at"],
                    "origin": "reconstructed"})
    for x in m.rows("SELECT x.id, x.experiment_id, x.attempt, x.exit_code, x.timed_out, x.duration_s, x.created_at "
                    "FROM executions x JOIN experiments e ON e.id=x.experiment_id WHERE e.run_id=?", rid):
        ok = x["exit_code"] == 0 and not x["timed_out"]
        out.append({"id": f"x{x['id']}", "phase": "execute", "experiment_id": x["experiment_id"],
                    "attempt": x["attempt"], "worker": None, "status": "ok" if ok else "failed",
                    "summary": (f"exit 0 in {x['duration_s']:.1f}s" if ok else
                                "timed out" if x["timed_out"] else f"exit {x['exit_code']}"),
                    "detail": {"execution_id": x["id"]}, "started_at": x["created_at"] - (x["duration_s"] or 0),
                    "finished_at": x["created_at"], "origin": "reconstructed"})
    for e in m.rows("SELECT * FROM experiments WHERE run_id=? AND finished_at IS NOT NULL", rid):
        if e["status"] == "completed":
            imp = e["improvement"] or 0.0
            summary = ("baseline" if e["is_baseline"] else ("SUCCESS" if e["success"] else "no improvement")) + \
                f": val {e['val_score']:.4f} ({imp:+.4f} vs baseline)"
        else:
            summary = f"{e['status']}: {(e['error'] or '').splitlines()[0][:200] if e['error'] else ''}"
        out.append({"id": f"e{e['id']}", "phase": "decide", "experiment_id": e["id"], "attempt": e["attempts"],
                    "worker": None, "status": {"completed": "ok", "interrupted": "interrupted"}.get(e["status"], "failed"),
                    "summary": summary, "detail": {"outcome": e["status"]}, "started_at": e["started_at"],
                    "finished_at": e["finished_at"], "origin": "reconstructed"})
    return sorted(out, key=lambda s: (s["finished_at"] or 0, str(s["id"])))


def state(m: Memory, run: dict, steps: list[dict]) -> dict:
    running = [s for s in steps if s["status"] == "running"]
    last = max((s["finished_at"] or s["started_at"] or 0 for s in steps), default=run["created_at"] or 0)
    stops = [s for s in steps if s["phase"] == "stop"]
    in_flight = m.one("SELECT COUNT(*) n FROM experiments WHERE run_id=? AND status='running'", run["id"])["n"]
    if running or in_flight:
        label = "running"
    elif stops:
        label = "stopped"
    else:
        label = "not running"
    return {"label": label, "running_steps": len(running), "in_flight_experiments": in_flight,
            "last_activity": last, "seconds_since_activity": round(time.time() - last, 1) if last else None,
            "stop_reason": stops[-1]["summary"] if stops else None}


def step_detail(m: Memory, run: dict, step_id: str) -> dict:
    """Everything behind one step: exact prompts/responses, code (+ previous version), execution logs."""
    tl = timeline(m, run)
    s = next((x for x in tl["steps"] if str(x["id"]) == str(step_id)), None)
    if s is None:
        raise LookupError(f"no step {step_id}")
    d = s["detail"]
    out = {"step": s, "llm_calls": [], "code": None, "previous_code": None, "execution": None, "hypotheses": []}
    call_ids = list(d.get("llm_call_ids") or ([d["llm_call_id"]] if d.get("llm_call_id") else []))
    role = {v: k for k, v in ROLE_PHASE.items()}.get(s["phase"])
    if role and s.get("started_at") and s.get("finished_at"):
        # Also any re-asks after invalid output inside this step's time window.
        extra = m.rows("SELECT id FROM llm_calls WHERE run_id=? AND role=? AND created_at BETWEEN ? AND ? "
                       "AND (experiment_id IS ? OR experiment_id = ?)", run["id"], role, s["started_at"],
                       s["finished_at"] + 0.5, s["experiment_id"], s["experiment_id"])
        call_ids = sorted(set(call_ids) | {r["id"] for r in extra})
    for cid in call_ids:
        c = m.one("SELECT id, role, provider, model, system, prompt, response, input_tokens, output_tokens, "
                  "cost_usd, error, created_at FROM llm_calls WHERE id=?", cid)
        if c:
            out["llm_calls"].append(c)
    if s["phase"] in ("implement", "execute") and s.get("experiment_id"):
        cv = None
        if d.get("code_version_id"):
            cv = m.one("SELECT * FROM code_versions WHERE id=?", d["code_version_id"])
        elif s.get("attempt"):
            cv = m.one("SELECT * FROM code_versions WHERE experiment_id=? AND attempt=?", s["experiment_id"], s["attempt"])
        elif call_ids:
            cv = m.one("SELECT * FROM code_versions WHERE llm_call_id=?", call_ids[0])
        if cv:
            out["code"] = cv
            if cv["parent_version_id"]:
                out["previous_code"] = m.one("SELECT * FROM code_versions WHERE id=?", cv["parent_version_id"])
    if s["phase"] == "execute":
        xid = d.get("execution_id")
        out["execution"] = (m.one("SELECT * FROM executions WHERE id=?", xid) if xid else None)
    if s["phase"] == "propose":
        ids = d.get("hypothesis_ids") or [h["id"] for h in m.hypotheses(run["id"]) if h["llm_call_id"] in call_ids]
        out["hypotheses"] = [h for h in m.hypotheses(run["id"]) if h["id"] in ids]
    return out


# --- checks ---------------------------------------------------------------------
def checks(m: Memory, run: dict, work_dir: Path | None, domain) -> list[dict]:
    """Audit of one run. Each check: {id, label, status: ok|warn|fail|info, detail}."""
    rid = run["id"]
    cfg = json.loads(run.get("config_json") or "{}")
    out = []

    def add(cid, label, status, detail):
        out.append({"id": cid, "label": label, "status": status, "detail": detail})

    # 1. Evaluation integrity: the trusted scorer, data preparation and held-out labels are unchanged.
    if domain is None:
        add("integrity", "Evaluation integrity", "warn", f"domain {run['domain']!r} not found; cannot verify")
    elif not work_dir or not (work_dir / "data" / "private").exists():
        add("integrity", "Evaluation integrity", "warn", "run workspace not found; cannot verify the held-out labels")
    else:
        try:
            domain.verify(work_dir / "data" / "private", json.loads(run["integrity_json"] or "{}"))
            n = len(json.loads(run["integrity_json"] or "{}"))
            add("integrity", "Evaluation integrity", "ok",
                f"{n} hashed files (scorer, prepare, baseline, domain spec, held-out labels) match the run start")
        except Exception as e:
            add("integrity", "Evaluation integrity", "fail", str(e))

    # 2. Budget.
    spent = m.total_cost(rid)
    budget = cfg.get("budget_usd")
    provider = (cfg.get("llm") or {}).get("provider")
    note = " (API-equivalent estimate; billed to the Claude subscription)" if provider == "claude-cli" else ""
    if budget:
        frac = spent / budget
        add("budget", "Budget", "warn" if frac > 0.9 else "ok", f"${spent:.4f} of ${budget:.2f} ({frac:.0%}){note}")
    else:
        add("budget", "Budget", "info", f"${spent:.4f} spent, no budget recorded{note}")

    # 3. LLM call health.
    calls = m.one("SELECT COUNT(*) n, SUM(error IS NOT NULL) e, SUM(error LIKE 'invalid output%') inv "
                  "FROM llm_calls WHERE run_id=?", rid)
    n, errs, inv = calls["n"] or 0, calls["e"] or 0, calls["inv"] or 0
    provider_errs = errs - inv
    add("llm", "LLM calls", "warn" if provider_errs else "ok",
        f"{n} calls; {inv} re-asked after schema-invalid output; {provider_errs} provider error(s)")

    # 4. Human approvals for raised limits.
    appr = m.rows("SELECT item, old_value, new_value, approved_by, created_at FROM approvals WHERE run_id=? "
                  "ORDER BY id", rid)
    add("approvals", "Approvals", "info",
        "; ".join(f"{a['item']}: {a['old_value']} -> {a['new_value']} by {a['approved_by']}" for a in appr)
        or "no limit was ever raised")

    # 5. Permission requests from the coder: recorded, never granted automatically.
    perms = m.rows("SELECT request, status FROM permission_requests WHERE run_id=?", rid)
    pending = [p for p in perms if p["status"] == "pending"]
    add("permissions", "Permission requests", "warn" if pending else "ok",
        (f"{len(pending)} pending (not granted): " + "; ".join(sorted({p['request'] for p in pending})))
        if pending else "none requested")

    # 6. Sandbox isolation.
    sb = cfg.get("sandbox") or {}
    net = sb.get("network", "none")
    add("sandbox", "Sandbox", "ok" if net == "none" else "warn",
        f"network={net}, memory={sb.get('memory')}, cpus={sb.get('cpus')}, timeout={sb.get('timeout_s')}s")

    # 7. Experiment outcomes, failures and repairs.
    exps = m.rows("SELECT * FROM experiments WHERE run_id=? AND is_baseline=0", rid)
    by = {k: sum(1 for e in exps if e["status"] == k) for k in ("completed", "failed", "interrupted", "running")}
    repaired = sum(1 for e in exps if (e["attempts"] or 0) > 1)
    status = "warn" if exps and by["failed"] / len(exps) > 0.5 else "ok"
    add("experiments", "Experiments", status,
        f"{len(exps)} of max {cfg.get('max_experiments')}: {by['completed']} completed, {by['failed']} failed, "
        f"{by['interrupted']} interrupted, {by['running']} running; {repaired} needed code repairs")

    # 8. Validation vs held-out test for the best result (agents never see test scores).
    done = [e for e in exps if e["status"] == "completed" and e["val_score"] is not None]
    if done and domain is not None:
        best = max(done, key=lambda e: e["val_score"] if domain.higher_is_better else -e["val_score"])
        gap = (best["test_score"] - best["val_score"]) * (1 if domain.higher_is_better else -1)
        nval = _count_lines(work_dir / "data" / "private" / "val_y.csv") if work_dir else None
        se = ((best["val_score"] * (1 - best["val_score"]) / nval) ** 0.5
              if nval and 0 <= best["val_score"] <= 1 else None)
        worse = se is not None and gap < -2 * se * 2 ** 0.5
        add("val_test", "Validation vs held-out test", "warn" if worse else "ok",
            f"best by validation #{best['id']}: val {best['val_score']:.4f}, test {best['test_score']:.4f} "
            f"(gap {gap:+.4f}{f'; noise ~{2 * se * 2 ** 0.5:.4f}' if se else ''})"
            + ("; possible overfitting to validation" if worse else ""))
    else:
        add("val_test", "Validation vs held-out test", "info", "no completed experiments yet")

    # 9. Reproducibility: same system code and sandbox image throughout.
    revs = {r["system_revision"] for r in m.rows("SELECT system_revision FROM experiments WHERE run_id=?", rid)}
    imgs = {r["image_id"] for r in m.rows("SELECT x.image_id FROM executions x JOIN experiments e ON "
                                          "e.id=x.experiment_id WHERE e.run_id=?", rid)}
    add("reproducibility", "Reproducibility", "warn" if len(revs) > 1 or len(imgs) > 1 else "ok",
        f"{len(revs)} system revision(s) ({', '.join(sorted(r or '?' for r in revs))}); {len(imgs)} sandbox image(s)"
        + ("; the system changed during the run, so compare results across revisions with care" if len(revs) > 1
           else ""))
    return out


def _count_lines(p: Path) -> int | None:
    try:
        return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return None
