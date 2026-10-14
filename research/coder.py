"""Coder: implements a hypothesis as a single sandboxed Python program, and repairs it on failure."""
from __future__ import annotations

from . import schemas
from .domain import Domain
from .llm import LLM

SYSTEM = """You implement machine-learning experiments as a single, complete Python program.
Follow the task contract exactly: read only the stated inputs, write exactly the required
outputs, seed all randomness from the SEED environment variable, and stay well inside
the time and memory limits. There is no network and no package installation; use only the
listed libraries. Return the full program, not a diff. If the hypothesis truly needs a
capability the sandbox lacks, list it in requested_permissions (a human decides) and
implement the closest feasible version anyway."""

MAX_CODE_CHARS = 60_000


class Coder:
    def __init__(self, llm: LLM, domain: Domain, timeout_s: int):
        self.llm, self.domain, self.timeout_s = llm, domain, timeout_s

    def prompt(self, ctx: dict) -> str:
        h = ctx["hypothesis"]
        parts = [f"TASK CONTRACT (wall-clock limit {self.timeout_s}s):\n{self.domain.contract.strip()}",
                 f"\nHYPOTHESIS: {h['title']}\nRATIONALE: {h['rationale']}\nMETHOD: {h['method']}"]
        if ctx.get("previous_code"):
            parts += [f"\nATTEMPT {ctx['attempt']}: the previous implementation failed.",
                      f"FAILURE:\n{ctx['previous_error']}",
                      f"PREVIOUS CODE:\n```python\n{ctx['previous_code']}\n```",
                      "Fix the problem (simplify or speed up if it timed out) and return the full program."]
        return "\n".join(parts)

    async def implement(self, *, experiment_id: int, hypothesis: dict, attempt: int,
                        previous_code: str | None = None, previous_error: str | None = None):
        ctx = {"hypothesis": {k: hypothesis[k] for k in ("title", "rationale", "method")},
               "attempt": attempt, "previous_code": previous_code, "previous_error": previous_error,
               "timeout_s": self.timeout_s}

        def check(data):
            code = data["code"]
            if len(code) > MAX_CODE_CHARS:
                raise ValueError(f"program too long ({len(code)} chars)")
            # Syntax check only: compile() parses, it does not execute anything.
            try:
                compile(code, "experiment.py", "exec")
            except SyntaxError as e:
                raise ValueError(f"program does not parse: {e}")

        data, call_id = await self.llm.structured(
            role="coder", system=SYSTEM, prompt=self.prompt(ctx), schema=schemas.CODE, schema_name="experiment_code",
            context=ctx, experiment_id=experiment_id, check=check)
        return data["code"], data["explanation"], data["requested_permissions"], call_id
