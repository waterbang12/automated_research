"""Controller: owns the loop. Sets up / resumes a run, enforces limits, schedules workers.

Per experiment: claim hypothesis -> coder writes program -> sandbox executes ->
trusted scorer -> (repair and retry on failure) -> LLM analysis -> record.

Every step is also written to `agent_steps` as it starts and ends (phase, status, timing,
worker, and links to the LLM call / code version / execution behind it), so a run can be
followed live and audited afterwards as one agent's trace.
When the hypothesis queue is empty, the researcher is asked for new ones using
everything recorded so far, which is what closes the feedback loop.
"""
from __future__ import annotations

import asyncio
import contextlib
import getpass
import hashlib
import json
import logging
import shutil
from pathlib import Path

from .coder import Coder
from .config import Config, expansions, guarded_values
from .domain import Domain, IntegrityError
from .evaluator import Evaluator
from .executor import DockerExecutor, ExecResult
from .llm import LLM, Budget, BudgetExceeded, LLMOutputError, ProviderUnavailable, credentials_missing
from .memory import Memory
from .researcher import Researcher

log = logging.getLogger("research")

PKG_DIR = Path(__file__).resolve().parent


class ApprovalRequired(RuntimeError):
    pass


def system_revision() -> str:
    h = hashlib.sha256()
    for p in sorted(PKG_DIR.glob("*.py")):
        h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()[:12]


def _seed(run_name: str, exp_id: int) -> int:
    return int(hashlib.sha256(f"{run_name}:{exp_id}".encode()).hexdigest()[:8], 16)


class Controller:
    def __init__(self, cfg: Config, *, domain: Domain | None = None, memory: Memory | None = None,
                 executor=None, adapters: dict | None = None, approve: bool = False):
        self.cfg = cfg
        self.domain = domain or Domain(cfg.domain)
        self.memory = memory or Memory(cfg.db_path)
        self.executor = executor or DockerExecutor(cfg.sandbox)
        self.adapters, self.approve = adapters, approve
        self.work = cfg.work_path
        self.public_dir, self.private_dir = self.work / "data" / "public", self.work / "data" / "private"
        self.revision = system_revision()
        self.stop_reason: str | None = None
        self.claim_lock = asyncio.Lock()
        self.empty_rounds = 0

    @contextlib.contextmanager
    def step(self, phase: str, **kw):
        """Record one agent step. The body may set st["status"], st["summary"] and st["detail"]."""
        sid = self.memory.start_step(self.run_id, phase, **kw)
        st = {"status": "ok", "summary": None, "detail": {}}
        try:
            yield st
        except (asyncio.CancelledError, KeyboardInterrupt):
            self.memory.finish_step(sid, "interrupted", st["summary"], **st["detail"])
            raise
        except BaseException as e:
            self.memory.finish_step(sid, "failed", f"{type(e).__name__}: {e}"[:500], **st["detail"])
            raise
        else:
            self.memory.finish_step(sid, st["status"], st["summary"], **st["detail"])

    # --- setup -----------------------------------------------------------
    async def setup(self) -> None:
        if self.adapters is None:
            missing = credentials_missing(self.cfg)
            if missing:
                raise RuntimeError(f"missing credentials: set {', '.join(missing)} "
                                   "(or use provider = \"offline\" for a credential-free run)")
        self.image_id = await self.executor.preflight()
        run = self.memory.get_run(self.cfg.name)
        limits = guarded_values(self.cfg)
        grow = expansions(json.loads(run["limits_json"]) if run else None, limits)
        if grow and not self.approve:
            items = "; ".join(f"{k}: {old} -> {new}" for k, old, new in grow)
            raise ApprovalRequired(f"these limits/permissions grow and need human approval: {items}. "
                                   "Re-run with --approve to confirm.")
        n = 0
        if run is None:
            await self._prepare_data()
            integrity = self.domain.snapshot(self.private_dir)
            objective = self.cfg.objective or self.domain.objective
            self.run_id = self.memory.create_run(self.cfg.name, self.domain.name, objective, self.cfg.to_dict(),
                                                 limits, self.revision, integrity)
            log.info("created run %r (id %d)", self.cfg.name, self.run_id)
        else:
            if run["domain"] != self.domain.name:
                raise RuntimeError(f"run {self.cfg.name!r} belongs to domain {run['domain']!r}")
            self.run_id = run["id"]
            integrity = json.loads(run["integrity_json"])
            self.domain.verify(self.private_dir, integrity)   # detect tampering between sessions
            self.memory.update_run_limits(self.run_id, self.cfg.to_dict(), limits)
            n = self.memory.recover_interrupted(self.run_id)
            log.info("resuming run %r (id %d); %d interrupted experiment(s) re-queued", self.cfg.name, self.run_id, n)
            objective = run["objective"]
        for k, old, new in grow:
            self.memory.record_approval(self.run_id, k, old, new, getpass.getuser())
        with self.step("session") as st:
            st["summary"] = (f"resumed run; {n} interrupted experiment(s) re-queued" if run else "created run")
            st["detail"] = {"system_revision": self.revision, "image_id": self.image_id, "limits": limits,
                            "approved": [{"item": k, "old": o, "new": v} for k, o, v in grow],
                            "provider": self.cfg.llm.provider, "model": self.cfg.llm.model}
        self.budget = Budget(self.memory, self.run_id, self.cfg.budget_usd)
        self.llm = LLM(self.cfg, self.memory, self.run_id, self.budget, adapters=self.adapters,
                       fixture=self.domain.offline_fixture())
        self.researcher = Researcher(self.llm, self.memory, self.domain, self.run_id, objective,
                                     self.cfg.sandbox.timeout_s)
        self.coder = Coder(self.llm, self.domain, self.cfg.sandbox.timeout_s)
        self.evaluator = Evaluator(self.domain, self.private_dir, integrity,
                                   self.llm if self.cfg.llm_analysis else None)

    async def _prepare_data(self) -> None:
        if (self.private_dir / "test_y.csv").exists():
            return
        for d in (self.public_dir, self.private_dir):
            d.mkdir(parents=True, exist_ok=True)
        code_dir = self.work / "data" / "prepare"
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / "prepare.py").write_text(self.domain.prepare_code(), encoding="utf-8")
        res = await self.executor.run(
            label="prepare", argv=["python", "/code/prepare.py"], env={},
            mounts=[(code_dir, "/code", True), (self.public_dir, "/public", False), (self.private_dir, "/private", False)])
        if not res.ok:
            raise RuntimeError(f"data preparation failed:\n{res.stderr}")
        log.info("prepared data: %s", res.stdout.strip())

    # --- main loop -------------------------------------------------------
    async def run(self) -> str:
        await self.setup()
        if self.memory.baseline(self.run_id) is None:
            await self.run_baseline()
        workers = [asyncio.create_task(self.worker(i)) for i in range(self.cfg.workers)]
        # A crashed worker sets stop_reason, so the others finish their current experiment and exit.
        errors = [r for r in await asyncio.gather(*workers, return_exceptions=True) if isinstance(r, BaseException)]
        reason = self.stop_reason or "done"
        with self.step("stop") as st:
            st["status"] = "failed" if errors else "ok"
            st["summary"] = reason
            st["detail"] = {"spent_usd": round(self.budget.spent(), 4), "budget_usd": self.cfg.budget_usd,
                            "experiments": self.memory.count_experiments(self.run_id)}
        if errors:
            raise errors[0]
        log.info("stopped: %s | spent $%.4f of $%.2f", reason, self.budget.spent(), self.cfg.budget_usd)
        return reason

    def stop(self, reason: str) -> None:
        if not self.stop_reason:
            self.stop_reason = reason
            log.info("stopping: %s", reason)

    async def worker(self, wid: int) -> None:
        try:
            await self._worker(wid)
        except BaseException as e:
            self.stop(f"worker {wid} crashed: {type(e).__name__}: {e}")
            raise

    async def _worker(self, wid: int) -> None:
        while True:
            async with self.claim_lock:
                if self.stop_reason:
                    return
                if self.memory.count_experiments(self.run_id) >= self.cfg.max_experiments:
                    self.stop(f"reached max_experiments={self.cfg.max_experiments}")
                    return
                h = self.memory.claim_next_hypothesis(self.run_id)
                if h is None:
                    try:
                        with self.step("propose", worker=wid) as st:
                            ids = await self.researcher.propose(self.cfg.hypotheses_per_round)
                            titles = [self.memory.one("SELECT title FROM hypotheses WHERE id=?", i)["title"]
                                      for i in ids]
                            st["summary"] = (f"proposed {len(ids)}: " + "; ".join(titles)) if ids else \
                                "no new hypotheses"
                            st["detail"] = {"hypothesis_ids": ids}
                    except BudgetExceeded as e:
                        self.stop(f"budget: {e}")
                        return
                    except ProviderUnavailable as e:
                        self.stop(f"provider: {e}")
                        return
                    except LLMOutputError as e:
                        log.warning("researcher failed: %s", e)
                        ids = []
                    log.info("[w%d] researcher proposed %d hypotheses: %s", wid, len(ids),
                             [self.memory.one("SELECT title FROM hypotheses WHERE id=?", i)["title"] for i in ids])
                    self.empty_rounds = 0 if ids else self.empty_rounds + 1
                    if self.empty_rounds >= 3:
                        self.stop("researcher produced no new hypotheses three times in a row")
                        return
                    continue
                exp_id = self.memory.start_experiment(self.run_id, h["id"], 0, self.revision)
                self.memory.update_experiment(exp_id, seed=_seed(self.cfg.name, exp_id))
            log.info("[w%d] exp #%d: %s", wid, exp_id, h["title"])
            try:
                await self.run_experiment(exp_id, h, wid)
            except IntegrityError as e:
                self.memory.finish_experiment(exp_id, status="failed", error=str(e))
                self.stop(f"INTEGRITY: {e}")
                return

    async def run_baseline(self) -> None:
        exp_id = self.memory.start_experiment(self.run_id, None, 0, self.revision, is_baseline=True)
        seed = _seed(self.cfg.name, exp_id)
        code = self.domain.baseline_code()
        cv = self.memory.add_code_version(experiment_id=exp_id, attempt=1, parent_version_id=None, code=code,
                                          code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                                          explanation="domain baseline", llm_call_id=None)
        res, out = await self._execute(exp_id, 1, cv, code, seed)
        if not res.ok:
            self.memory.finish_experiment(exp_id, status="failed", error=res.stderr[-2000:])
            raise RuntimeError(f"baseline failed:\n{res.stderr[-2000:]}")
        with self.step("score", experiment_id=exp_id, attempt=1) as st:
            scores = self.evaluator.score(out)
            st["summary"] = f"baseline val {self.domain.metric} {scores['val']:.4f}"
            st["detail"] = {"val": scores["val"], "test": scores["test"]}
        self.memory.finish_experiment(exp_id, status="completed", seed=seed, attempts=1, code_version_id=cv,
                                      val_score=scores["val"], test_score=scores["test"], baseline_val=scores["val"],
                                      improvement=0.0, success=0)
        log.info("baseline: val %s = %.4f", self.domain.metric, scores["val"])

    async def _execute(self, exp_id: int, attempt: int, cv_id: int, code: str, seed: int,
                       wid: int | None = None) -> tuple[ExecResult, Path]:
        ws = self.work / f"exp_{exp_id:04d}" / f"attempt_{attempt}"
        if ws.exists():
            shutil.rmtree(ws)
        (ws / "code").mkdir(parents=True)
        (ws / "out").mkdir()
        (ws / "code" / "experiment.py").write_text(code, encoding="utf-8")
        with self.step("execute", experiment_id=exp_id, attempt=attempt, worker=wid) as st:
            res = await self.executor.run(
                label=f"{self.run_id}-{exp_id}-{attempt}", argv=["python", "/code/experiment.py"],
                env={"SEED": str(seed), "PYTHONHASHSEED": "0"},
                mounts=[(ws / "code", "/code", True), (self.public_dir, "/data", True), (ws / "out", "/out", False)])
            xid = self.memory.add_execution(experiment_id=exp_id, code_version_id=cv_id, attempt=attempt,
                                            image=self.cfg.sandbox.image, image_id=res.image_id,
                                            command=json.dumps(res.command), exit_code=res.exit_code,
                                            timed_out=int(res.timed_out), duration_s=res.duration_s,
                                            stdout_tail=res.stdout, stderr_tail=res.stderr)
            st["status"] = "ok" if res.ok else "failed"
            st["summary"] = (f"exit 0 in {res.duration_s:.1f}s" if res.ok else
                             f"timed out after {self.cfg.sandbox.timeout_s}s" if res.timed_out else
                             f"exit {res.exit_code} after {res.duration_s:.1f}s")
            st["detail"] = {"execution_id": xid, "code_version_id": cv_id, "workspace": str(ws)}
        (ws / "stdout.txt").write_text(res.stdout, encoding="utf-8")
        (ws / "stderr.txt").write_text(res.stderr, encoding="utf-8")
        return res, ws / "out"

    async def run_experiment(self, exp_id: int, h: dict, wid: int | None = None) -> None:
        seed = self.memory.one("SELECT seed FROM experiments WHERE id=?", exp_id)["seed"]
        base = self.memory.baseline(self.run_id)
        prev_code = prev_err = prev_cv = None
        attempts = self.cfg.max_fix_attempts + 1
        ids = {"experiment_id": exp_id, "worker": wid}
        for attempt in range(1, attempts + 1):
            try:
                with self.step("implement", hypothesis_id=h["id"], attempt=attempt, **ids) as st:
                    code, expl, perms, call_id = await self.coder.implement(
                        experiment_id=exp_id, hypothesis=h, attempt=attempt, previous_code=prev_code,
                        previous_error=prev_err)
                    st["summary"] = (f"repair after: {prev_err.splitlines()[0][:120]}" if prev_err else "first version") \
                        + f" | {expl[:200]}"
                    st["detail"] = {"llm_call_id": call_id, "requested_permissions": perms}
            except BudgetExceeded as e:
                # Not the hypothesis's fault: leave it queued for a resumed run with more budget.
                self.memory.finish_experiment(exp_id, status="interrupted", attempts=attempt - 1, error=f"budget: {e}")
                self.memory.set_hypothesis_status(h["id"], "pending")
                self._decide(exp_id, h, wid, "interrupted", f"budget: {e}")
                self.stop(f"budget: {e}")
                return
            except ProviderUnavailable as e:
                self.memory.finish_experiment(exp_id, status="interrupted", attempts=attempt - 1, error=f"provider: {e}")
                self.memory.set_hypothesis_status(h["id"], "pending")
                self._decide(exp_id, h, wid, "interrupted", f"provider: {e}")
                self.stop(f"provider: {e}")
                return
            except LLMOutputError as e:
                prev_err = f"coder produced no valid program: {e}"
                break
            for p in perms:
                self.memory.add_permission_request(self.run_id, exp_id, p)
                log.warning("exp #%d requested permission %r (recorded; not granted)", exp_id, p)
            cv = self.memory.add_code_version(experiment_id=exp_id, attempt=attempt, parent_version_id=prev_cv,
                                              code=code, code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                                              explanation=expl, llm_call_id=call_id)
            self.memory.update_experiment(exp_id, attempts=attempt, code_version_id=cv)
            res, out = await self._execute(exp_id, attempt, cv, code, seed, wid)
            prev_code, prev_cv = code, cv
            if not res.ok:
                why = f"timed out after {self.cfg.sandbox.timeout_s}s" if res.timed_out else f"exit code {res.exit_code}"
                prev_err = f"{why}\nSTDERR (tail):\n{res.stderr[-3000:]}\nSTDOUT (tail):\n{res.stdout[-1000:]}"
                log.info("exp #%d attempt %d failed: %s", exp_id, attempt, why)
                continue
            try:
                with self.step("score", attempt=attempt, **ids) as st:
                    scores = self.evaluator.score(out)
                    st["summary"] = f"val {self.domain.metric} {scores['val']:.4f}"
                    st["detail"] = {"val": scores["val"], "test": scores["test"]}
            except IntegrityError:
                raise
            except Exception as e:
                if not self.evaluator.is_score_error(e):
                    raise
                prev_err = f"program exited 0 but its outputs were rejected by the scorer: {e}"
                log.info("exp #%d attempt %d: outputs rejected: %s", exp_id, attempt, e)
                continue
            await self._record_success(exp_id, h, base, scores, res, wid)
            return
        self.memory.finish_experiment(exp_id, status="failed", error=prev_err)
        self.memory.set_hypothesis_status(h["id"], "done")
        self._decide(exp_id, h, wid, "failed", f"failed after {attempt} attempt(s): {(prev_err or '').splitlines()[0][:200]}")
        log.info("exp #%d FAILED after %d attempt(s)", exp_id, attempt)

    def _decide(self, exp_id: int, h: dict, wid: int | None, outcome: str, summary: str, **detail) -> None:
        """The deterministic outcome of an experiment, as the last step of its trace."""
        with self.step("decide", experiment_id=exp_id, hypothesis_id=h["id"], worker=wid) as st:
            st["status"] = {"failed": "failed", "interrupted": "interrupted"}.get(outcome, "ok")
            st["summary"] = summary
            st["detail"] = {"outcome": outcome, **detail}

    async def _record_success(self, exp_id: int, h: dict, base: dict, scores: dict, res: ExecResult,
                              wid: int | None = None) -> None:
        imp, success = self.evaluator.judge(scores["val"], base["val_score"])
        prior = [e["val_score"] for e in self.memory.experiments(self.run_id)
                 if e["status"] == "completed" and e["id"] != exp_id]
        best_before = (max(prior) if self.domain.higher_is_better else min(prior)) if prior else None
        verdict = analysis = None
        lessons: list[str] = []
        if self.evaluator.llm is not None:
            try:
                with self.step("reflect", experiment_id=exp_id, hypothesis_id=h["id"], worker=wid) as st:
                    a = await self.evaluator.analyse(experiment_id=exp_id, hypothesis=h, val=scores["val"],
                                                     baseline_val=base["val_score"], improvement=imp, success=success,
                                                     stdout=res.stdout, best_so_far=best_before)
                    verdict, analysis, lessons = a["verdict"], a["analysis"], a["lessons"]
                    st["summary"] = f"analyst: {verdict} | {analysis[:200]}"
                    st["detail"] = {"verdict": verdict, "lessons": lessons}
            except BudgetExceeded as e:
                self.stop(f"budget: {e}")
            except ProviderUnavailable as e:
                self.stop(f"provider: {e}")
            except LLMOutputError as e:
                log.warning("analysis failed for exp #%d: %s", exp_id, e)
        # Test score is stored for the final report only; agents only ever see validation scores.
        self.memory.finish_experiment(exp_id, status="completed", val_score=scores["val"], test_score=scores["test"],
                                      baseline_val=base["val_score"], improvement=imp, success=int(success),
                                      verdict=verdict, analysis=analysis, lessons=json.dumps(lessons))
        self.memory.set_hypothesis_status(h["id"], "done")
        self._decide(exp_id, h, wid, "success" if success else "no_improvement",
                     f"{'SUCCESS' if success else 'no improvement'}: val {scores['val']:.4f} ({imp:+.4f} vs baseline, "
                     f"threshold {self.domain.min_improvement})", val=scores["val"], improvement=imp,
                     best_before=best_before)
        log.info("exp #%d: val %.4f (%+.4f vs baseline) %s", exp_id, scores["val"], imp,
                 "SUCCESS" if success else "no improvement")
