"""Evaluator: deterministic scoring with the domain's trusted scorer, plus optional LLM analysis.

The decision of whether a hypothesis succeeded is purely numeric (improvement
over the baseline's validation score >= the domain threshold). The LLM verdict
is recorded as commentary and feeds the next research round; it never overrides it.
"""
from __future__ import annotations

from pathlib import Path

from . import schemas
from .domain import Domain
from .llm import LLM

SYSTEM = """You review one completed experiment from an automated research loop. Given the
hypothesis, its measured validation score and the baseline, state whether the evidence
supports the hypothesis and extract short, reusable lessons that should guide the next
experiments. Be specific and quantitative; do not speculate beyond the evidence."""


class Evaluator:
    def __init__(self, domain: Domain, private_dir: Path, integrity: dict[str, str], llm: LLM | None):
        self.domain, self.private_dir, self.integrity, self.llm = domain, private_dir, integrity, llm
        self.scorer = domain.scorer()

    def score(self, out_dir: Path) -> dict:
        """Raises IntegrityError if evaluation assets changed, scorer.ScoreError if outputs are invalid."""
        self.domain.verify(self.private_dir, self.integrity)
        return self.scorer.score(out_dir, self.private_dir)

    def is_score_error(self, e: Exception) -> bool:
        return isinstance(e, self.scorer.ScoreError)

    def judge(self, val: float, baseline_val: float) -> tuple[float, bool]:
        imp = self.domain.improvement(val, baseline_val)
        return imp, imp >= self.domain.min_improvement

    async def analyse(self, *, experiment_id: int, hypothesis: dict, val: float, baseline_val: float,
                      improvement: float, success: bool, stdout: str, best_so_far: float | None) -> dict:
        ctx = {"hypothesis": {k: hypothesis[k] for k in ("title", "rationale", "method", "expected_metric")},
               "val_score": val, "baseline_val": baseline_val, "improvement": improvement, "success": success,
               "min_improvement": self.domain.min_improvement, "best_so_far": best_so_far,
               "stdout_tail": stdout[-2000:]}
        prompt = (f"HYPOTHESIS: {hypothesis['title']}\nRATIONALE: {hypothesis['rationale']}\n"
                  f"METHOD: {hypothesis['method']}\nPREDICTED: {hypothesis['expected_metric']}\n\n"
                  f"MEASURED validation {self.domain.metric}: {val:.4f}\nBASELINE: {baseline_val:.4f}\n"
                  f"IMPROVEMENT: {improvement:+.4f} (success threshold {self.domain.min_improvement}; "
                  f"deterministic outcome: {'success' if success else 'no success'})\n"
                  f"BEST BEFORE THIS EXPERIMENT: {best_so_far}\n\nPROGRAM OUTPUT (tail):\n{stdout[-2000:]}")
        data, _ = await self.llm.structured(role="analyst", system=SYSTEM, prompt=prompt, schema=schemas.ANALYSIS,
                                            schema_name="analysis", context=ctx, experiment_id=experiment_id)
        return data
