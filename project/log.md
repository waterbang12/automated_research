# Log: what we tried and what happened

*Kind of content: history. Each entry says what was tried and its outcome, with evidence: a run
name (local `runs/<name>.db`; `runs/` is gitignored), a commit, a test, or a committed sample.
Add new entries at the end. Do not rewrite old entries; correct them with a new dated entry.*

## 2026-10-14

- **Claude Code subscription as a provider.** Added `claude-cli` (`claude -p --json-schema`, tools off). The first real run failed: Claude Code 2.1.160 rejected `claude-opus-5-5` ("2.1.280 or newer is required").
  - **Hang bug:** the error left the run hung, because a crashed worker blocked `asyncio.run` from shutting down. Fixed so the run stops cleanly with the reason (D6). After `claude update` to 2.1.299, a 2-experiment smoke run succeeded.
  - Evidence: `run:smoke-claude-cli` (its checks still show the 5 provider errors and the mid-run code change); commit `929fc96`.
- **digits-demo** (8 experiments, Claude): every hypothesis beat the logistic baseline (0.9592). The best was the SVC+kNN ensemble at 0.9889 val, 0.9796 test. The cheap kNN and the heavy SVM were within noise of each other. Evidence: `run:digits-demo`.
- **digits-cheap-vs-heavy** (6 experiments, Claude): asked directly whether cheap models match an RBF-SVM.
  - **kNN vs SVM:** tuned kNN tied the RBF-SVM (val 0.9889 each; paired McNemar p = 1, disagreeing on 4 of 539 examples).
  - **Weak branches:** naive Bayes and shallow trees came in at −0.0167 vs the baseline but were not significant (p = 0.136). Linear models showed no detectable gain.
  - Evidence: `run:digits-cheap-vs-heavy`, `docs/sample-graph/`.
- **Research graph, phase 1** (commit `da9b01c`). Problems found while building it, all fixed:
  - Paths to the question shortcut through assumptions and superseded conclusions.
  - Method titles came from the first sentence of the method ("Load train.csv.").
  - **Overclaiming:** 1-NN was labelled "contradicted" at z = 0.16. This led to `not_supported` vs `contradicted` and the paired McNemar test (D8).
  - When the researcher's own prediction held but the hypothesis missed the fixed "beat the baseline" rule, the graph now says so.
- **Agent step log and Run page** (commit `db4eecc`). Verified on a live run (steps streaming, then "stopped"), on reconstructed traces of older runs, and on a fake run with repairs, a timeout and a scorer rejection. Evidence: `test:tests/test_steps.py::*`, `run:offline-trace`.

## 2026-10-14 to 2026-10-15

- **Phase 2 and problem formulation** (commit `9d8ffd6`):
  - **Added:** live decision records, the replay slider, `new-domain`, `check-domain`, and `/raw` data mounts.
  - **planted** (6 experiments, Claude, est. $1.07): RMSE went from 2.337 (ridge baseline) to 0.524 val / 0.542 test against a known floor of about 0.5, closing **99%** of the gap.
  - **Structure found:** by round 2 the decision record named the planted structure ("additive … plus a strong x2·x3 interaction"; group "at most through an intercept").
  - **What it missed:** it did not establish that x4 is a threshold or x1 a sine, and listed that as uncertain.
  - Evidence: `run:planted`, `docs/sample-planted/`, `docs/formulating-and-testing.md`.
- **Bugs the planted run exposed, both fixed:**
  - The run report and the trend line took the highest score as best even for lower-is-better metrics, so the report named the worst model.
  - RMSE comparisons used a proportion formula. They now use a paired t-test on per-example squared errors.
  - Evidence: `test:tests/test_units.py::test_report_best_respects_lower_is_better`, `test:tests/test_graph.py::test_paired_t`.

## 2026-10-15

- **Codex CLI provider.**
  - Installed `@openai/codex` 0.162.1 in a scratch directory only, to read `codex exec --help`.
  - **Real-CLI check:** called the real CLI without signing in. The JSONL events were `thread.started`, `turn.started`, `error`, then `turn.failed` with 401.
  - **Field names:** the success-event names the adapter reads (`turn.completed`, `agent_message`, `cached_input_tokens`) were confirmed as strings in the bundled `codex.exe`, not from a successful run.
  - **What works:** the adapter's real-CLI call launched correctly through `node codex.js`, and the 401 surfaced as `ProviderUnavailable`.
  - **Not yet done:** a signed-in run (roadmap N1, questions Q1). Evidence: `test:tests/test_codex_cli.py::*`.
- **Project metadata:** `AGENTS.md`, `CLAUDE.md` (imports AGENTS.md), and `project/`, with machine-checked evidence references (`tests/test_project_metadata.py`). Related work was verified against six primary sources.
