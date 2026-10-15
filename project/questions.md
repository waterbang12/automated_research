# Open questions

*Kind of content: uncertainties. These are things we do not know yet, with what would settle
them. When one is answered, record the answer and its evidence here, then move the fact into
`capabilities.yaml` or `state.md`.*

| # | Question | Why it matters | What would answer it |
|---|---|---|---|
| Q1 | With a signed-in Codex CLI, does `--output-schema` produce schema-valid JSON reliably? Do `turn.completed` usage fields match what the adapter reads (`input_tokens`, `cached_input_tokens`, `output_tokens`)? Which model names and `model_reasoning_effort` values are accepted? | The adapter was tested against a fake and an unauthenticated real CLI only. | Roadmap N1: a 2-experiment signed-in run, then inspect `llm_calls`. |
| Q2 | Do the live Anthropic and OpenAI APIs accept the exact requests the adapters send? For Anthropic, that includes the server-side fallback beta together with `output_config`. | Both adapters have only been tested against a local fake HTTP server. | Roadmap N3. |
| Q3 | How should the margin δ be chosen for equivalence, ablation and null claims (M5)? Per domain, per claim, or derived from noise? | Too small a δ makes claims fail on noise; too large a δ makes them vacuous. | Design spike on the planted domain, where the true effects are known. |
| Q4 | How should many comparisons within a run be handled (multiple testing)? | The "winner" of many experiments is partly luck. There is currently no correction. | Decide alongside M5/M6, e.g. Holm correction for claims, and repeated CV for the final ranking. |
| Q5 | Can the researcher's decision records be checked for faithfulness, i.e. does the stated reasoning match what was done? | They are self-reports (D12). | Compare stated `evidence_used` and claims against the graph's evidence. Typed claims (M5) make part of this mechanical. |
| Q6 | For literature retrieval (M7): which sources are permitted, and does retrieval run on the host (with network) or in the sandbox (without)? | The sandbox has no network by design (D3). | An owner decision before M7. |
| Q7 | Is one planted problem representative of the agent's ability? | One run is one sample (log.md). | Roadmap N4, plus more planted designs: a noise-only target, known Bayes error, leakage traps. |
| Q8 | Does the dagre layout stay usable for runs with hundreds of experiments? | The viewer re-lays out the whole graph on every filter change. | Generate a large fake run and time it; consider incremental layout. |
