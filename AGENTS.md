# AGENTS.md

Instructions for AI coding agents (Codex, Claude Code, others) working in this repository.
Claude Code reads this file through `CLAUDE.md`, so keep everything here, not there.

**automated_research** is an autonomous empirical research loop. A researcher, coder and
analyst (phases of one agent) propose hypotheses, write programs that run in a Docker sandbox,
and have them scored by a trusted scorer they cannot change. Every step, decision and result is
persisted in SQLite and shown as an auditable trace and a reasoning graph.

## Start of every session

Read these, in this order, before changing anything:

| Question | File |
|---|---|
| What are we building, and why? | [project/vision.md](project/vision.md) |
| What exists now, and how does it fit together? | [project/state.md](project/state.md) |
| What is verified to work, and what is only proposed? | [project/capabilities.yaml](project/capabilities.yaml) |
| What have we tried, and what happened? | [project/log.md](project/log.md) |
| What should be done next, and how will we know it is done? | [project/roadmap.yaml](project/roadmap.yaml) (`next_actions`, milestone `acceptance_criteria`) |
| Why is it built this way? | [project/decisions.md](project/decisions.md) |
| What is still unknown? | [project/questions.md](project/questions.md) |
| What related systems and papers exist? | [project/related-work.yaml](project/related-work.yaml) |

User-facing documentation lives in `README.md` and `docs/`. The `project/` files describe the
*project* (state, plans, evidence), not how to use it. Link to the docs instead of copying them.

## Commands (Windows, from the repo root, PowerShell or the VS Code terminal)

```powershell
.venv\Scripts\python -m pytest -q                          # all tests (Docker tests skip if Docker is down)
.venv\Scripts\python -m research build-sandbox             # once: builds autoresearch-sandbox:1
.venv\Scripts\python -m research check-domain <name>       # no-LLM check of a problem definition
.venv\Scripts\python -m research run configs/<cfg>.toml    # start or resume a run
.venv\Scripts\python -m research graph serve               # viewer at http://127.0.0.1:8765/
```

## Rules

- **Evidence before state.** Change `project/` files only for things you verified in this session: a passing test, a real run, a command's output, or a primary source for related work. Mark everything else as `proposed` or put it in `questions.md`. Never upgrade a capability's status without adding the evidence. `tests/test_project_metadata.py` checks that evidence references exist.
- **At the end of a task**, append to `project/log.md` (what was tried, and the outcome with evidence), update `capabilities.yaml` and `roadmap.yaml` only where the evidence supports it, and add to `decisions.md` if a design choice was made.
- **Keep the existing integrations working:** the Anthropic API, OpenAI API, Claude CLI, Codex CLI and offline providers. Run the full test suite before committing.
- **Integrity of evaluation:**
  - Agents never see validation or test labels.
  - The scorer and data preparation are trusted code; generated programs never run on the host.
  - Never edit a domain's `scorer.py`, `prepare.py`, `baseline.py`, `domain.toml` or data while a run that uses it might be resumed. Integrity hashes will stop the run.
- **Epistemic care** (enforced in `research/graph.py`, tested in `tests/test_graph.py`):
  - Weak evidence is never phrased as a refutation.
  - `rules_out` and `epistemic="formal"` are never derived automatically.
  - Rejected branches are kept, never deleted.
- **Not yet:** do not add a new agent framework, knowledge-graph service, separate dashboard or vector database without an evaluation showing it is needed (`project/decisions.md`, D13).
- **Style:** match the surrounding code. Plain Python, asyncio, stdlib where possible, no new dependencies without a reason. Commit messages end with the attribution lines your tool requires.
