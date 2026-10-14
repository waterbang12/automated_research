import asyncio
import json

import pytest

from research.controller import ApprovalRequired, Controller
from research.domain import Domain, IntegrityError
from research.memory import Memory

from fakes import BASELINE_ACC, FakeExecutor, ScriptedAdapter, make_config, patch_baseline


def run(cfg, adapter, executor=None, approve=False, memory=None):
    ctl = Controller(cfg, domain=patch_baseline(Domain("digits")), memory=memory or Memory(cfg.db_path),
                     executor=executor or FakeExecutor(), adapters={"anthropic": adapter}, approve=approve)
    reason = asyncio.run(ctl.run())
    return ctl, reason


def test_full_loop_records_everything_and_feeds_back(tmp_path):
    cfg = make_config(tmp_path, max_experiments=5)
    adapter = ScriptedAdapter({"idea 1": ["acc=0.9"], "idea 2": ["acc=0.6"], "idea 3": ["crash", "acc=0.75"]})
    ctl, reason = run(cfg, adapter)
    assert "max_experiments=5" in reason
    exps = ctl.memory.experiments(ctl.run_id)
    base = [e for e in exps if e["is_baseline"]][0]
    assert base["val_score"] == pytest.approx(BASELINE_ACC)
    real = [e for e in exps if not e["is_baseline"]]
    assert len(real) == 5 and all(e["status"] == "completed" for e in real)
    e1 = next(e for e in real if e["title"] == "idea 1")
    assert e1["success"] == 1 and e1["improvement"] == pytest.approx(0.2)
    assert next(e for e in real if e["title"] == "idea 2")["success"] == 0
    # Repaired experiment keeps both code versions, linked, and both executions.
    e3 = next(e for e in real if e["title"] == "idea 3")
    cvs = ctl.memory.rows("SELECT * FROM code_versions WHERE experiment_id=? ORDER BY attempt", e3["id"])
    assert [c["attempt"] for c in cvs] == [1, 2] and cvs[1]["parent_version_id"] == cvs[0]["id"]
    assert len(ctl.memory.rows("SELECT * FROM executions WHERE experiment_id=?", e3["id"])) == 2
    coder_ctx = [c for c in adapter.contexts["coder"] if c["hypothesis"]["title"] == "idea 3"]
    assert "boom" in coder_ctx[1]["previous_error"]
    # Feedback: later researcher calls saw earlier results; parent ids were sanitised.
    seen = [len(c["experiments"]) for c in adapter.contexts["researcher"]]
    assert seen[0] == 0 and max(seen) >= 2
    for h in ctl.memory.hypotheses(ctl.run_id):
        assert 9999 not in json.loads(h["parent_experiment_ids"])
    # Every experiment has a seed and a code revision recorded.
    assert all(e["seed"] and e["system_revision"] for e in exps)


def test_failed_experiments_are_preserved(tmp_path):
    cfg = make_config(tmp_path, max_experiments=2, workers=1)
    _, _ = run(cfg, ScriptedAdapter({"idea 1": ["timeout"], "idea 2": ["badoutput"]}))
    mem = Memory(cfg.db_path)
    exps = {e["title"]: e for e in mem.experiments(1) if not e["is_baseline"]}
    assert exps["idea 1"]["status"] == "failed" and "timed out" in exps["idea 1"]["error"]
    assert exps["idea 2"]["status"] == "failed" and "rejected by the scorer" in exps["idea 2"]["error"]
    assert exps["idea 1"]["attempts"] == 2


def test_budget_stops_run_and_keeps_completed_work(tmp_path):
    # Each call costs (1000*1 + 1000*2)/1e6 = $0.003 actual; worst case reserves max_tokens of output.
    cfg = make_config(tmp_path, budget_usd=0.1, max_experiments=50)
    cfg.llm.max_tokens = 4000
    ctl, reason = run(cfg, ScriptedAdapter(cost_tokens=(1000, 1000)))
    assert reason.startswith("budget")
    assert ctl.budget.spent() <= cfg.budget_usd
    done = [e for e in ctl.memory.experiments(ctl.run_id) if e["status"] == "completed" and not e["is_baseline"]]
    assert len(done) >= 1


def test_resume_and_limit_expansion_needs_approval(tmp_path):
    cfg = make_config(tmp_path, max_experiments=2)
    ctl, _ = run(cfg, ScriptedAdapter())
    # Simulate a crash mid-experiment.
    mem = ctl.memory
    h = mem.add_hypothesis(ctl.run_id, {"title": "stuck", "rationale": "r", "method": "m", "expected_metric": 0,
                                        "priority": 1.0, "parent_experiment_ids": []}, None)
    mem.set_hypothesis_status(h, "claimed")
    stuck = mem.start_experiment(ctl.run_id, h, 1, "rev")
    before = {e["id"]: e["val_score"] for e in mem.experiments(ctl.run_id) if e["status"] == "completed"}

    cfg.max_experiments = 4
    with pytest.raises(ApprovalRequired):
        run(cfg, ScriptedAdapter(), memory=mem)
    ctl2, _ = run(cfg, ScriptedAdapter(), memory=mem, approve=True)
    after = {e["id"]: e for e in mem.experiments(ctl2.run_id)}
    assert all(after[i]["val_score"] == v for i, v in before.items())       # nothing lost
    assert after[stuck]["status"] == "interrupted"
    assert mem.count_experiments(ctl2.run_id) == 4
    assert any(e["title"] == "stuck" and e["status"] == "completed" for e in after.values())
    appr = mem.rows("SELECT * FROM approvals")
    assert appr[0]["item"] == "max_experiments" and appr[0]["new_value"] == "4"
    # Lowering limits needs no approval.
    cfg.max_experiments = 1
    run(cfg, ScriptedAdapter(), memory=mem)


def test_network_access_requires_approval(tmp_path):
    cfg = make_config(tmp_path)
    cfg.sandbox.network = "bridge"
    with pytest.raises(ApprovalRequired):
        run(cfg, ScriptedAdapter())


def test_permission_requests_recorded_not_granted(tmp_path):
    cfg = make_config(tmp_path, max_experiments=1, workers=1)
    ctl, _ = run(cfg, ScriptedAdapter({"idea 1": ["acc=0.99"]}))
    reqs = ctl.memory.rows("SELECT * FROM permission_requests")
    assert [(r["request"], r["status"]) for r in reqs] == [("network", "pending")]


def test_tampering_with_labels_is_detected(tmp_path):
    cfg = make_config(tmp_path, max_experiments=1, workers=1)
    ctl, _ = run(cfg, ScriptedAdapter())
    (ctl.private_dir / "val_y.csv").write_text("0\n" * 100)
    cfg.max_experiments = 1
    with pytest.raises(IntegrityError):
        run(cfg, ScriptedAdapter(), memory=ctl.memory)


def test_integrity_checked_before_each_score(tmp_path):
    cfg = make_config(tmp_path, max_experiments=3, workers=1)

    class TamperingExecutor(FakeExecutor):
        async def run(self, **kw):
            res = await super().run(**kw)
            if self.calls.count("x") == 0 and kw["label"].endswith("-2-1"):
                self.calls.append("x")
                priv = [src for src, dst, _ in kw["mounts"] if dst == "/data"][0].parent / "private"
                (priv / "test_y.csv").write_text("1\n" * 100)
            return res

    ctl, reason = run(cfg, ScriptedAdapter(), executor=TamperingExecutor())
    assert reason.startswith("INTEGRITY")
