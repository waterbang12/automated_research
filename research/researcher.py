"""Researcher: turns the recorded history into new, prioritised, testable hypotheses."""
from __future__ import annotations

import json
import re

from . import schemas
from .domain import Domain
from .llm import LLM
from .memory import Memory

SYSTEM = """You lead an automated empirical research loop. Each hypothesis you write is
implemented by a separate coding agent and executed in an offline sandbox, then scored
by a fixed, trusted evaluator you cannot change. Propose hypotheses that are:
- concrete enough to implement in one short Python program within the time limit;
- informed by the recorded results: build on what worked, avoid what failed, explain why;
- genuinely different from every experiment already run or queued.
Prefer ideas with a high expected gain per experiment. Cite the experiment ids that
motivated each hypothesis in parent_experiment_ids.

Before the hypotheses, record your decision for this round, as a researcher would in a lab
notebook: what the results so far support (cite #ids; say so when the evidence is weak or a
difference is within noise), which directions you considered but are NOT proposing now and why,
which assumptions you rely on or now doubt, and what remains uncertain. Only claim what the
recorded results show; a method that underperformed once is not proven useless. For each
hypothesis, state in falsified_if the measured result that would count against it."""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


class Researcher:
    def __init__(self, llm: LLM, memory: Memory, domain: Domain, run_id: int, objective: str,
                 timeout_s: int):
        self.llm, self.memory, self.domain, self.run_id = llm, memory, domain, run_id
        self.objective, self.timeout_s = objective, timeout_s

    def context(self, n: int) -> dict:
        base = self.memory.baseline(self.run_id)
        exps = [e for e in self.memory.experiments(self.run_id)
                if not e["is_baseline"] and e["status"] in ("completed", "failed")]
        pending = [h for h in self.memory.hypotheses(self.run_id) if h["status"] in ("pending", "claimed")]
        return {
            "n": n, "objective": self.objective, "metric": self.domain.metric,
            "higher_is_better": self.domain.higher_is_better,
            "baseline": {"id": base["id"], "val_score": base["val_score"]},
            "experiments": [{
                "id": e["id"], "title": e["title"], "method": e["method"], "status": e["status"],
                "val_score": e["val_score"], "improvement": e["improvement"], "verdict": e["verdict"],
                "error": (e["error"] or "")[-600:], "lessons": json.loads(e["lessons"] or "[]"),
                "parents": json.loads(e["parent_experiment_ids"] or "[]"),
            } for e in exps],
            "pending": [{"title": h["title"], "method": h["method"]} for h in pending],
        }

    def prompt(self, ctx: dict) -> str:
        better = "higher" if ctx["higher_is_better"] else "lower"
        lines = [f"OBJECTIVE: {ctx['objective']}",
                 f"METRIC: validation {ctx['metric']} ({better} is better). Held-out test scores are never shown.",
                 f"TIME LIMIT PER EXPERIMENT: {self.timeout_s}s",
                 f"\nTASK CONTRACT GIVEN TO THE CODER:\n{self.domain.contract.strip()}",
                 f"\nBASELINE (experiment #{ctx['baseline']['id']}): {ctx['baseline']['val_score']:.4f}"]
        done = ctx["experiments"]
        if done:
            ranked = sorted([e for e in done if e["val_score"] is not None],
                            key=lambda e: e["val_score"], reverse=ctx["higher_is_better"])
            lines.append("\nBEST SO FAR:")
            lines += [f"  #{e['id']} {e['val_score']:.4f} ({e['improvement']:+.4f}) {e['title']}" for e in ranked[:5]]
            lines.append("\nALL EXPERIMENTS (oldest first):")
            for e in done:
                score = f"{e['val_score']:.4f} ({e['improvement']:+.4f})" if e["val_score"] is not None else "no score"
                lines.append(f"- #{e['id']} [{e['status']}] {score} :: {e['title']}")
                lines.append(f"    method: {e['method'][:400]}")
                if e["error"]:
                    lines.append(f"    error: {e['error'][-300:]}")
                for lesson in e["lessons"][:3]:
                    lines.append(f"    lesson: {lesson}")
        else:
            lines.append("\nNo experiments beyond the baseline yet.")
        if ctx["pending"]:
            lines.append("\nALREADY QUEUED (do not duplicate):")
            lines += [f"- {h['title']}" for h in ctx["pending"]]
        lines.append(f"\nPropose exactly {ctx['n']} new hypotheses. priority is in [0, 1]; "
                     "expected_metric is your predicted validation score.")
        return "\n".join(lines)

    async def propose(self, n: int) -> list[int]:
        ctx = self.context(n)
        valid_ids = {ctx["baseline"]["id"], *(e["id"] for e in ctx["experiments"])}

        def check(data):
            if not data["hypotheses"]:
                raise ValueError("no hypotheses returned")
            for h in data["hypotheses"]:
                if not h["title"].strip() or not h["method"].strip():
                    raise ValueError("empty title or method")

        data, call_id = await self.llm.structured(
            role="researcher", system=SYSTEM, prompt=self.prompt(ctx), schema=schemas.HYPOTHESES,
            schema_name="hypotheses", context=ctx, check=check)
        known = {_norm(h["title"]) for h in self.memory.hypotheses(self.run_id)}
        d = data["decision"]
        d["evidence_used"] = [i for i in d["evidence_used"] if i in valid_ids]
        for k, cap in (("alternatives_considered", 4), ("assumptions", 3), ("uncertain", 3)):
            d[k] = d[k][:cap]
        self.last_decision = d
        self.last_decision_id = self.memory.add_decision(self.run_id, call_id, d)
        ids = []
        for h in data["hypotheses"][:n]:
            if _norm(h["title"]) in known:
                continue                       # exact repeat of an earlier idea
            known.add(_norm(h["title"]))
            h["priority"] = min(1.0, max(0.0, float(h["priority"])))
            h["parent_experiment_ids"] = [i for i in h["parent_experiment_ids"] if i in valid_ids]
            ids.append(self.memory.add_hypothesis(self.run_id, h, call_id))
        return ids
