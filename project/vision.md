# Vision

*Kind of content: intent. This file is the owner's goals and the reasons for them, not claims
about the code. For what exists, see [state.md](state.md) and [capabilities.yaml](capabilities.yaml).*

## What we are building

An autonomous research agent that runs empirical investigations unattended. It proposes
hypotheses, writes and runs experiments in a sandbox, and has the results scored by code it
cannot change. Its runs can be **audited** and its conclusions **traced to evidence**.

The central question the system must always be able to answer about a run:

> How did we arrive at this conclusion, which alternatives did we consider, and precisely what
> evidence caused us to narrow or change direction?

## Why

- Autonomous ML-experiment agents already exist (see [related-work.yaml](related-work.yaml)). They mostly report *what* scored best, not *why* a direction was taken or how strong the evidence is.
- An agent's conclusions are only useful if a person can check them. That needs:
  - **trusted evaluation:** fixed scorers, held-out data, integrity checks
  - **visible reasoning:** a recorded decision for each step
  - **honest evidence labels:** weak results are not presented as refutations
- The owner wants to *use* this for research questions, not only to optimise a metric.

## Direction (long term)

1. **From optimizer to hypothesis tester.** Today the trusted evaluator checks only "did this beat the baseline?". The goal is that it checks the claim a hypothesis actually makes: improvement, equivalence, ablation ("x matters"), direction of effect, or no effect. See `eval.typed_claims` in `capabilities.yaml`.
2. **Reliable evidence:** repeated cross-validation and multiple seeds, so open questions such as "is A really better than B?" can be closed.
3. **Literature-informed method selection:** method cards from verified sources, with provenance, and cheap pilots before expensive methods.
4. **Graph-aware memory:** retrieval and prioritisation of past results across runs, without treating negative results as low-value.
5. **Provider-agnostic:** any capable model through an API or a local CLI (Claude Code and Codex today), with honest cost accounting.

## Non-goals (for now)

These are deliberately excluded, at the owner's instruction (2026-10-15; see
[decisions.md](decisions.md), D13):
- a new agent framework, a knowledge-graph service, a separate dashboard, or a vector database
- browser-side control of runs (the Run page is read-only)
- multi-GPU or long-running training (the sandbox is one CPU, about 2 minutes per program)

## Principles

- Evidence over narrative. Self-reported reasoning is shown next to the measured evidence, never instead of it.
- Keep every branch. Rejected, failed and unexplored directions are part of the record.
- Small, testable increments. Each milestone has acceptance criteria and tests.
