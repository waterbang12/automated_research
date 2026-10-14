import asyncio
from pathlib import Path

import pytest

from research import schemas
from research.config import expansions, guarded_values, load
from research.domain import Domain
from research.llm import LLM, Budget, BudgetExceeded, LLMOutputError
from research.memory import Memory

from fakes import ScriptedAdapter, make_config

ROOT = Path(__file__).resolve().parent.parent


def test_validator():
    good = {"code": "x", "explanation": "y", "requested_permissions": []}
    schemas.validate(good, schemas.CODE)
    for bad in ({"code": "x"}, {**good, "extra": 1}, {**good, "requested_permissions": [1]}, {**good, "code": 3}):
        with pytest.raises(schemas.ValidationError):
            schemas.validate(bad, schemas.CODE)
    with pytest.raises(schemas.ValidationError):
        schemas.validate({"verdict": "maybe", "analysis": "", "lessons": []}, schemas.ANALYSIS)
    with pytest.raises(schemas.ValidationError):   # bool is not a number
        schemas.validate({"hypotheses": [{"title": "t", "rationale": "r", "method": "m", "expected_metric": True,
                                          "priority": 1, "parent_experiment_ids": []}]}, schemas.HYPOTHESES)


def test_schemas_are_strict_everywhere():
    def walk(s):
        if s.get("type") == "object":
            assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
            for v in s["properties"].values():
                walk(v)
        if s.get("type") == "array":
            walk(s["items"])
    for s in (schemas.HYPOTHESES, schemas.CODE, schemas.ANALYSIS):
        walk(s)


def test_example_configs_load():
    for p in (ROOT / "configs").glob("*.toml"):
        cfg = load(p)
        assert cfg.sandbox.network == "none"


def test_expansions():
    cfg = make_config(Path("."))
    old = guarded_values(cfg)
    cfg.budget_usd = 5
    cfg.sandbox.memory = "2g"
    grown = {k for k, *_ in expansions(old, guarded_values(cfg))}
    assert grown == {"budget_usd", "sandbox.memory"}
    cfg.budget_usd = 0.5
    cfg.sandbox.memory = "1g"
    assert expansions(old, guarded_values(cfg)) == []


def _llm(tmp_path, budget=1.0, **kw):
    cfg = make_config(tmp_path, **kw)
    mem = Memory(tmp_path / "db.sqlite")
    adapter = ScriptedAdapter(cost_tokens=(1000, 500))
    return cfg, mem, LLM(cfg, mem, 1, Budget(mem, 1, budget), adapters={"anthropic": adapter}), adapter


def test_llm_retries_invalid_output_and_records_cost(tmp_path):
    cfg, mem, llm, adapter = _llm(tmp_path)
    adapter.invalid_left = 1
    data, call_id = asyncio.run(llm.structured(role="analyst", system="s", prompt="p", schema=schemas.ANALYSIS,
                                               schema_name="a", context={}))
    assert data["verdict"] == "inconclusive"
    calls = mem.rows("SELECT * FROM llm_calls ORDER BY id")
    assert len(calls) == 2 and calls[0]["error"].startswith("invalid output") and calls[1]["error"] is None
    assert calls[1]["cost_usd"] == pytest.approx((1000 * 1 + 500 * 2) / 1e6)
    assert "rejected by validation" in calls[1]["prompt"]


def test_llm_gives_up_after_retry_limit(tmp_path):
    cfg, mem, llm, adapter = _llm(tmp_path)
    adapter.invalid_left = 99
    with pytest.raises(LLMOutputError):
        asyncio.run(llm.structured(role="analyst", system="s", prompt="p", schema=schemas.ANALYSIS,
                                   schema_name="a", context={}))
    assert len(mem.rows("SELECT * FROM llm_calls")) == cfg.llm_validation_retries + 1


def test_budget_reserves_worst_case(tmp_path):
    cfg, mem, llm, _ = _llm(tmp_path, budget=0.01)
    cfg.llm.max_tokens = 16000      # worst case 16000 * $2/M = $0.032 > $0.01
    with pytest.raises(BudgetExceeded):
        asyncio.run(llm.structured(role="analyst", system="s", prompt="p", schema=schemas.ANALYSIS,
                                   schema_name="a", context={}))
    assert mem.rows("SELECT * FROM llm_calls") == []


def test_concurrent_reservations_cannot_overspend(tmp_path):
    mem = Memory(tmp_path / "db.sqlite")
    b = Budget(mem, 1, 1.0)

    async def go():
        results = await asyncio.gather(*(b.reserve(0.3) for _ in range(5)), return_exceptions=True)
        return sum(isinstance(r, BudgetExceeded) for r in results)
    assert asyncio.run(go()) == 2


def test_unknown_model_billed_at_max_rate(tmp_path):
    cfg, mem, llm, _ = _llm(tmp_path)
    cfg.pricing = {"cheap": (1.0, 1.0), "dear": (10.0, 50.0)}
    assert llm.price("mystery", 1_000_000, 0) == 10.0


def test_digits_scorer_rejects_bad_outputs(tmp_path):
    scorer = Domain("digits").scorer()
    priv, out = tmp_path / "p", tmp_path / "o"
    priv.mkdir(), out.mkdir()
    for s in ("val", "test"):
        (priv / f"{s}_y.csv").write_text("1\n2\n3\n")
    (out / "val_pred.csv").write_text("1\n2\n3\n")
    (out / "test_pred.csv").write_text("1\n2\n")
    with pytest.raises(scorer.ScoreError, match="expected 3 rows"):
        scorer.score(out, priv)
    (out / "test_pred.csv").write_text("1\n2\n3.5\n")
    with pytest.raises(scorer.ScoreError):
        scorer.score(out, priv)
    (out / "test_pred.csv").write_text("1\n2\n4\n")
    assert scorer.score(out, priv) == {"val": 1.0, "test": pytest.approx(2 / 3)}


def test_offline_fixture_uses_history():
    fx = Domain("digits").offline_fixture()
    base = {"id": 1, "val_score": 0.95}
    first = fx.researcher({"n": 2, "baseline": base, "experiments": [], "pending": []})["hypotheses"]
    assert all(h["parent_experiment_ids"] == [1] for h in first)
    done = [{"id": 2, "title": first[0]["title"], "method": first[0]["method"], "status": "completed",
             "val_score": 0.98, "improvement": 0.03}]
    nxt = fx.researcher({"n": 2, "baseline": base, "experiments": done, "pending": []})["hypotheses"]
    assert all(h["parent_experiment_ids"] == [2] for h in nxt)
    assert {h["title"] for h in nxt}.isdisjoint({first[0]["title"]})
    code = fx.coder({"hypothesis": nxt[0]})["code"]
    compile(code, "x", "exec")


def test_report_best_respects_lower_is_better(tmp_path, capsys):
    from research.__main__ import print_report
    from research.memory import Memory
    m = Memory(tmp_path / "r.db")
    rid = m.create_run("r", "planted", "minimise rmse", {}, {}, "rev", {})
    for val, imp in ((2.0, 0.0), (0.9, 1.1), (0.6, 1.4)):          # rmse: lower is better, improvement positive
        e = m.start_experiment(rid, None, 0, "rev", is_baseline=(imp == 0.0))
        m.finish_experiment(e, status="completed", val_score=val, test_score=val, improvement=imp, success=int(imp > 0))
    print_report(m, "r")
    assert "Best by validation: #3" in capsys.readouterr().out
