"""Deterministic test doubles. The fake executor never runs generated code on the host:
programs carry a `# FAKE: <behaviour>` marker that tells it what outcome to simulate."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from research.config import Config, LLMConfig, SandboxConfig
from research.executor import ExecResult
from research.llm import Usage

N_VAL = N_TEST = 100


class FakeExecutor:
    def __init__(self):
        self.calls = []

    async def preflight(self):
        return "sha256:fake"

    async def run(self, *, label, mounts, env, argv, timeout_s=None):
        self.calls.append(label)
        await asyncio.sleep(0.01)
        m = {dst: Path(src) for src, dst, _ in mounts}
        if label == "prepare":
            (m["/public"] / "train.csv").write_text("f0,label\n1,1\n")
            for split, n in (("val", N_VAL), ("test", N_TEST)):
                (m["/private"] / f"{split}_y.csv").write_text("\n".join(str(i % 10) for i in range(n)) + "\n")
            return ExecResult(0, False, 0.0, "prepared", "", [], "sha256:fake")
        code = (m["/code"] / "experiment.py").read_text()
        behaviour = re.search(r"# FAKE: (\S+)", code).group(1)
        if behaviour == "crash":
            return ExecResult(1, False, 0.0, "", "Traceback: boom", [], "sha256:fake")
        if behaviour == "timeout":
            return ExecResult(None, True, 0.0, "", "", [], "sha256:fake")
        if behaviour == "badoutput":
            (m["/out"] / "val_pred.csv").write_text("not a label\n")
            return ExecResult(0, False, 0.0, "", "", [], "sha256:fake")
        acc = float(behaviour.split("=")[1])
        for split, n in (("val", N_VAL), ("test", N_TEST)):
            right = int(acc * n)
            preds = [i % 10 if i < right else (i + 1) % 10 for i in range(n)]
            (m["/out"] / f"{split}_pred.csv").write_text("\n".join(map(str, preds)) + "\n")
        return ExecResult(0, False, 0.0, "ok", "", [], "sha256:fake")


class ScriptedAdapter:
    """Fake provider. `plan` maps hypothesis title -> list of fake behaviours per coding attempt."""

    def __init__(self, plan: dict[str, list[str]] | None = None, cost_tokens=(0, 0), invalid_first: int = 0):
        self.plan = plan or {}
        self.contexts = {"researcher": [], "coder": [], "analyst": []}
        self.counter = 0
        self.cost_tokens = cost_tokens
        self.invalid_left = invalid_first

    async def complete(self, *, role, model, system, prompt, schema, schema_name, context):
        self.contexts[role].append(context)
        usage = Usage(*self.cost_tokens, model)
        if self.invalid_left:
            self.invalid_left -= 1
            return '{"not": "valid"}', usage
        if role == "researcher":
            hs = []
            parents = [e["id"] for e in context["experiments"]] or [context["baseline"]["id"]]
            for _ in range(context["n"]):
                self.counter += 1
                hs.append({"title": f"idea {self.counter}", "rationale": f"after {len(context['experiments'])} results",
                           "method": "m", "expected_metric": 0.9, "priority": 0.5,
                           "parent_experiment_ids": parents[-1:] + [9999], "falsified_if": "val below 0.85"})
            decision = {"current_belief": f"{len(context['experiments'])} results so far",
                        "evidence_used": parents[-1:] + [9999],
                        "alternatives_considered": [{"idea": f"alt after {len(context['experiments'])}",
                                                     "why_not_now": "lower expected gain"}],
                        "assumptions": ["validation is representative"],
                        "uncertain": [f"uncertainty {len(context['experiments'])}"],
                        "why_these": "cheapest informative tests"}
            return json.dumps({"decision": decision, "hypotheses": hs}), usage
        if role == "coder":
            title = context["hypothesis"]["title"]
            steps = self.plan.get(title, ["acc=0.8"])
            b = steps[min(context["attempt"] - 1, len(steps) - 1)]
            return json.dumps({"code": f"# FAKE: {b}\nprint('hi')\n", "explanation": "x",
                               "requested_permissions": ["network"] if b == "acc=0.99" else []}), usage
        return json.dumps({"verdict": "inconclusive", "analysis": "a", "lessons": ["l"]}), usage


def make_config(tmp: Path, **overrides) -> Config:
    cfg = Config(name="t", domain="digits", max_experiments=4, budget_usd=1.0, workers=2, hypotheses_per_round=2,
                 max_fix_attempts=1, llm=LLMConfig(provider="anthropic", model="m"),
                 sandbox=SandboxConfig(), pricing={"m": (1.0, 2.0)}, root=tmp)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


BASELINE_ACC = 0.7


def patch_baseline(domain):
    domain.baseline_code = lambda: f"# FAKE: acc={BASELINE_ACC}\n"
    return domain
