"""Agent step log (recorded live by the controller), reconstruction for older runs, step details,
run checks, and the read-only Run page API."""
import asyncio
import json
import threading
import urllib.request

import pytest

from research import steps as S
from research.controller import Controller
from research.domain import Domain
from research.llm import ProviderUnavailable
from research.memory import Memory

from fakes import FakeExecutor, ScriptedAdapter, make_config, patch_baseline

PLAN = {"idea 1": ["crash", "acc=0.8"],                 # repaired on attempt 2
        "idea 2": ["timeout", "badoutput", "crash"],    # fails all attempts
        "idea 3": ["acc=0.6"], "idea 4": ["acc=0.99"]}  # acc=0.99 also requests the network permission


def run_fake(tmp, adapter=None, **cfg_over):
    cfg = make_config(tmp, max_experiments=4, workers=1, max_fix_attempts=2, **cfg_over)
    ctl = Controller(cfg, domain=patch_baseline(Domain("digits")), memory=Memory(cfg.db_path),
                     executor=FakeExecutor(), adapters={"anthropic": adapter or ScriptedAdapter(PLAN)})
    asyncio.run(ctl.run())
    return cfg, ctl


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    return run_fake(tmp_path_factory.mktemp("steps"))


def by_exp(steps, title_or_id, ctl):
    if isinstance(title_or_id, str):
        title_or_id = next(e["id"] for e in ctl.memory.experiments(ctl.run_id) if e["title"] == title_or_id)
    return [(s["phase"], s["attempt"], s["status"]) for s in steps if s["experiment_id"] == title_or_id]


def test_trace_is_recorded_live_and_complete(recorded):
    cfg, ctl = recorded
    tl = S.timeline(ctl.memory, ctl.memory.get_run(cfg.name))
    steps = tl["steps"]
    assert tl["source"] == "recorded" and tl["state"]["label"] == "stopped"
    assert steps[0]["phase"] == "session" and steps[-1]["phase"] == "stop"
    assert "max_experiments=4" in steps[-1]["summary"]
    assert all(s["status"] != "running" and s["finished_at"] >= s["started_at"] for s in steps)
    assert [s["phase"] for s in steps if s["phase"] == "propose"] and all(
        s["worker"] == 0 for s in steps if s["phase"] in ("propose", "implement"))
    # Repair chain, exactly as it happened.
    assert by_exp(steps, "idea 1", ctl) == [
        ("implement", 1, "ok"), ("execute", 1, "failed"), ("implement", 2, "ok"), ("execute", 2, "ok"),
        ("score", 2, "ok"), ("reflect", None, "ok"), ("decide", None, "ok")]
    # Every failure mode of idea 2 is visible, ending in a failed decision.
    assert by_exp(steps, "idea 2", ctl) == [
        ("implement", 1, "ok"), ("execute", 1, "failed"), ("implement", 2, "ok"), ("execute", 2, "ok"),
        ("score", 2, "failed"), ("implement", 3, "ok"), ("execute", 3, "failed"), ("decide", None, "failed")]
    score_fail = next(s for s in steps if s["phase"] == "score" and s["status"] == "failed")
    assert "ScoreError" in score_fail["summary"]
    decide = next(s for s in steps if s["phase"] == "decide" and s["status"] == "ok")
    assert decide["detail"]["outcome"] in ("success", "no_improvement") and "val" in decide["detail"]


def test_step_detail_links_exact_prompt_code_and_logs(recorded):
    cfg, ctl = recorded
    run = ctl.memory.get_run(cfg.name)
    steps = S.timeline(ctl.memory, run)["steps"]
    repair = next(s for s in steps if s["phase"] == "implement" and s["attempt"] == 2 and
                  s["hypothesis_title"] == "idea 1")
    d = S.step_detail(ctl.memory, run, repair["id"])
    assert len(d["llm_calls"]) == 1 and "previous implementation failed" in d["llm_calls"][0]["prompt"]
    assert "boom" in d["llm_calls"][0]["prompt"]                       # the error the coder was shown
    assert d["code"]["attempt"] == 2 and d["previous_code"]["attempt"] == 1
    ex = next(s for s in steps if s["phase"] == "execute" and s["status"] == "failed")
    dx = S.step_detail(ctl.memory, run, ex["id"])
    assert dx["execution"] and dx["code"]
    prop = next(s for s in steps if s["phase"] == "propose")
    dp = S.step_detail(ctl.memory, run, prop["id"])
    assert [h["title"] for h in dp["hypotheses"]] == ["idea 1", "idea 2"] and dp["llm_calls"][0]["role"] == "researcher"
    assert prop["detail"]["decision"]["why_these"] == "cheapest informative tests"
    assert prop["detail"]["decision_id"]


def test_checks(recorded, tmp_path):
    cfg, ctl = recorded
    run = ctl.memory.get_run(cfg.name)
    c = {x["id"]: x for x in S.checks(ctl.memory, run, cfg.work_path, Domain("digits"))}
    assert c["integrity"]["status"] == "ok"
    assert c["permissions"]["status"] == "warn" and "network" in c["permissions"]["detail"]
    assert c["sandbox"]["status"] == "ok" and c["reproducibility"]["status"] == "ok"
    assert "1 failed" in c["experiments"]["detail"] and "needed code repairs" in c["experiments"]["detail"]
    assert set(c) == {"integrity", "budget", "llm", "approvals", "permissions", "sandbox", "experiments",
                      "val_test", "reproducibility"}
    # Tampering with a held-out label file is reported as a failed check.
    labels = cfg.work_path / "data" / "private" / "val_y.csv"
    original = labels.read_text()
    try:
        labels.write_text("0\n" * 100)
        c2 = {x["id"]: x for x in S.checks(ctl.memory, run, cfg.work_path, Domain("digits"))}
        assert c2["integrity"]["status"] == "fail"
    finally:
        labels.write_text(original)


def test_provider_outage_is_an_interrupted_step_not_a_failed_experiment(tmp_path):
    class Down(ScriptedAdapter):
        async def complete(self, *, role, **kw):
            if role == "coder":
                raise ProviderUnavailable("usage limit reached")
            return await super().complete(role=role, **kw)
    cfg, ctl = run_fake(tmp_path, Down())
    steps = S.timeline(ctl.memory, ctl.memory.get_run(cfg.name))["steps"]
    impl = next(s for s in steps if s["phase"] == "implement")
    assert impl["status"] == "failed" and "usage limit" in impl["summary"]
    dec = next(s for s in steps if s["phase"] == "decide")
    assert dec["status"] == "interrupted" and dec["detail"]["outcome"] == "interrupted"
    assert "provider" in steps[-1]["summary"] and steps[-1]["phase"] == "stop"


def test_resume_marks_dangling_steps_interrupted(recorded, tmp_path):
    cfg, ctl = recorded
    m = Memory(cfg.db_path)
    sid = m.start_step(ctl.run_id, "implement", experiment_id=None)
    m.recover_interrupted(ctl.run_id)
    assert m.one("SELECT status FROM agent_steps WHERE id=?", sid)["status"] == "interrupted"


def test_older_runs_get_a_reconstructed_trace(recorded, tmp_path):
    import sqlite3
    cfg, ctl = recorded
    db = tmp_path / "old.db"
    src, dst = sqlite3.connect(cfg.db_path), sqlite3.connect(db)
    src.backup(dst)
    src.close()
    dst.execute("DROP TABLE agent_steps")       # as written before step logging existed
    dst.commit()
    dst.close()
    m = Memory(db)                               # opening recreates an empty table; nothing else changes
    run = m.get_run(cfg.name)
    tl = S.timeline(m, run)
    assert tl["source"] == "reconstructed"
    phases = {s["phase"] for s in tl["steps"]}
    assert {"propose", "implement", "execute", "reflect", "decide"} <= phases
    assert all(s["origin"] == "reconstructed" for s in tl["steps"])
    prop = next(s for s in tl["steps"] if s["phase"] == "propose")
    assert prop["summary"].startswith("proposed 2: idea 1; idea 2")
    impl = next(s for s in tl["steps"] if s["phase"] == "implement")
    assert S.step_detail(m, run, impl["id"])["llm_calls"][0]["prompt"]


def test_run_page_api(recorded):
    from research.viewer_server import serve
    cfg, _ = recorded
    srv = serve(cfg.db_path.parent, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    get = lambda p: json.load(urllib.request.urlopen(base + p, timeout=30))
    try:
        q = f"db={cfg.db_path.name}&run=1"
        tl = get(f"/api/steps?{q}")
        assert tl["source"] == "recorded" and tl["steps"]
        d = get(f"/api/step?{q}&id={tl['steps'][4]['id']}")
        assert d["step"]["id"] == tl["steps"][4]["id"]
        assert {c["id"] for c in get(f"/api/checks?{q}")} >= {"integrity", "budget"}
    finally:
        srv.shutdown()
