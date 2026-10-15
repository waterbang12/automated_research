# Architectural decisions

*Kind of content: decisions that were made, and why. Add new entries at the end; mark replaced
decisions as superseded rather than deleting them. "Owner" means the decision was the
repository owner's instruction in a session.*

| # | Date | Decision | Why | Status |
|---|---|---|---|---|
| D1 | 2026-10-14 | Plain Python, asyncio, SQLite and Docker; call provider SDKs and CLIs directly; no agent framework. | Few moving parts, so every step is inspectable and testable. | active |
| D2 | 2026-10-14 | Success is decided by the trusted scorer's numbers (improvement ≥ `min_improvement`). The LLM analyst's verdict is commentary only. | An LLM must not be able to grade its own work. | active (M5 extends it to typed claims) |
| D3 | 2026-10-14 | Agents see only validation scores. Test scores are stored for humans. Domain files and labels are integrity-hashed. | Detects overfitting to validation; prevents tampering. | active |
| D4 | 2026-10-14 | Subscription CLIs are providers (`claude-cli`, then `codex-cli` on 2026-10-15). API keys are removed from the CLI's environment by default. | The owner prefers to use subscriptions instead of API keys. Stripping keys keeps billing predictable. | active |
| D5 | 2026-10-14 | CLI usage is budgeted as an *estimate*: Claude's `total_cost_usd`, Codex's tokens × `[pricing]`. It is never presented as billing. | The CLIs do not report charges in a form that equals API billing. | active |
| D6 | 2026-10-14 | A provider outage (auth, usage limit, outdated CLI, timeout) stops the run cleanly. The experiment is marked interrupted and its hypothesis re-queued. | Avoids burning retries and hanging. The run resumes with the same command. Found when an outdated CLI hung the run (log.md). | active |
| D7 | 2026-10-14 | The research graph is *derived* from existing records into additive tables, with deterministic ids. It never deletes; rows that are no longer produced become stale. | Works for every past run, cannot drift from the records, and keeps rejected branches. | active |
| D8 | 2026-10-14 | Epistemic rules: weak evidence is never a refutation; `contradicted` requires significance; `rules_out`/formal are never automatic; strength comes from paired tests on per-example predictions. | Owner spec: "do not turn weak empirical evidence into claims of logical impossibility". | active |
| D9 | 2026-10-14 | The viewer uses vendored UMD builds of React, React Flow, dagre and htm, served by a stdlib server. There is no npm or build step. | Nothing to install; works offline. The cost is no TypeScript. | active |
| D10 | 2026-10-14 | Researcher, coder and analyst stay as separate structured calls, presented and audited as one agent's trace. They are not merged into one long agent session. | Merging would serialise the workers, grow the context, and let the analyst grade its own idea. Owner: "agent-like … easily accessible and checkable runs". | active |
| D11 | 2026-10-14 | The Run page is read-only; runs are started and stopped from the terminal. | Owner decision. | active |
| D12 | 2026-10-14 | Round decisions are *recorded* self-reports from the researcher. Experiment decisions are *derived* from the scorer. Both are shown, labelled by provenance. | Captures intent at decision time without letting it override measured evidence. | active |
| D13 | 2026-10-15 | No new agent framework, knowledge-graph service, dashboard or vector database until an evaluation shows the need. | Owner instruction. | active |
| D14 | 2026-10-15 | Project knowledge lives in `AGENTS.md` (read by Codex; imported by Claude Code through `CLAUDE.md`) and `project/`. Facts need evidence references, which `tests/test_project_metadata.py` checks. | One source for every AI tool. Machine-checked evidence keeps the metadata from drifting into wishful claims. | active |
| D15 | 2026-10-15 | On Windows, the npm `codex.cmd` shim is resolved to `node <…>/@openai/codex/bin/codex.js` instead of being run through `cmd.exe`. The prompt goes over stdin and the schema through a file. | cmd.exe re-parses arguments, and command lines have length limits. | active |
