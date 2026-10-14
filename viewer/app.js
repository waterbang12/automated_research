/* Research Reasoning Graph viewer. Plain script: React + React Flow UMD globals, htm for JSX-like templates. */
(function () {
  "use strict";
  const { useState, useEffect, useMemo, useCallback, useRef } = React;
  const RF = window.ReactFlow;
  const html = htm.bind(React.createElement);

  const TYPES = ["QUESTION", "ASSUMPTION", "OBSERVATION", "HYPOTHESIS", "METHOD", "EXPERIMENT",
    "EVIDENCE", "DECISION", "CONCLUSION", "OPEN_QUESTION"];
  const DEFAULT_HIDDEN = new Set(["METHOD"]);

  // Status -> visual class + icon + label. Meaning never rests on colour alone.
  const STATUS = {
    supported: ["good", "✓", "supported"], current: ["good", "✓", "current"], adopt: ["good", "✓", "adopted"],
    supports: ["good", "✓", "supports"], consistent: ["good", "✓", "consistent"], completed: ["", "●", "completed"],
    contradicted: ["critical", "✕", "contradicted"], contradicts: ["critical", "✕", "contradicts"],
    reject: ["critical", "✕", "rejected (empirical)"],
    not_supported: ["warning", "~", "not supported"], set_aside: ["warning", "~", "set aside"],
    supported_not_best: ["faded", "↓", "beat baseline, not best"], deprioritize: ["faded", "↓", "deprioritized"],
    superseded: ["faded", "⤳", "superseded"],
    questioned: ["warning", "?", "questioned"], unverified: ["unexplored", "?", "unverified"],
    open: ["warning", "?", "open"], accepted: ["", "●", "accepted"], reference: ["", "◆", "reference"],
    invalid: ["invalid", "⚠", "invalid"], failed: ["invalid", "⚠", "failed"],
    untested_invalid: ["invalid", "⚠", "untested (invalid run)"], interrupted: ["invalid", "⏸", "interrupted"],
    not_investigated: ["unexplored", "○", "not investigated"], in_progress: ["unexplored", "◔", "in progress"],
    running: ["unexplored", "◔", "running"], stale: ["faded", "·", "stale"],
  };
  const st = (s) => STATUS[s] || ["", "●", s || ""];

  // Edge label read along the arrow (arrows always point downstream in the reasoning).
  const INVERSE = {
    tests: "tested by", supports: "supported by", contradicts: "contradicted by", refines: "refined by",
    supersedes: "superseded by", depends_on: "used by", inspired_by: "inspired", rules_out: "ruled out by",
    selects: "selected by", deprioritizes: "deprioritized by",
  };
  const edgeLabel = (e) => (e.rel === "via" ? "via " + e.via.join(", ").toLowerCase()
    : e.flow_src === e.src ? e.rel.replace("_", " ") : (INVERSE[e.rel] || e.rel));
  const EDGE_STYLE = {
    supports: { stroke: "var(--good)", strokeWidth: 2 },
    contradicts: { stroke: "var(--critical)", strokeWidth: 2, strokeDasharray: "6 4" },
    deprioritizes: { stroke: "var(--muted)", strokeDasharray: "4 4" },
    rules_out: { stroke: "var(--critical)", strokeWidth: 2 },
    selects: { stroke: "var(--good)", strokeWidth: 2 },
    supersedes: { stroke: "var(--muted)", strokeDasharray: "2 3" },
  };

  const api = (path, params) => {
    const q = new URLSearchParams(params).toString();
    return fetch(`/api/${path}?${q}`).then(async (r) => {
      const body = await r.json();
      if (!r.ok) throw new Error(body.error || r.statusText);
      return body;
    });
  };
  const fmt = (x, d = 4) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));
  const when = (t) => (t ? new Date(t * 1000).toLocaleString() : "–");

  // ---------------------------------------------------------------- layout
  function layout(nodes, edges) {
    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: "TB", nodesep: 28, ranksep: 70, marginx: 20, marginy: 20 });
    g.setDefaultEdgeLabel(() => ({}));
    nodes.forEach((n) => g.setNode(n.id, { width: 250, height: 92 }));
    edges.forEach((e) => g.setEdge(e.flow_src, e.flow_dst));
    dagre.layout(g);
    const pos = {};
    nodes.forEach((n) => { const p = g.node(n.id); pos[n.id] = { x: p.x - 125, y: p.y - 46 }; });
    return pos;
  }

  // ---------------------------------------------------------------- node card
  function RNode({ data }) {
    const n = data.node;
    const [cls, icon, label] = st(n.status);
    const classes = ["rnode", cls ? `st-${cls}` : "", data.match ? "match" : "", data.selected ? "selected" : "",
      data.dim ? "dim" : ""].join(" ");
    const badgeCls = ["good", "critical", "warning"].includes(cls) ? cls : "";
    return html`
      <div class=${classes} style=${{ "--stripe": `var(--t-${n.type.toLowerCase()})` }} title=${n.summary || n.title}>
        <${RF.Handle} type="target" position=${RF.Position.Top} style=${{ opacity: 0 }} />
        <div class="head"><span>${n.type.replace("_", " ")}</span>
          ${n.confidence ? html`<span title="evidence strength (approximate, one split)">${n.confidence}</span>` : null}</div>
        <div class="title">${n.title}</div>
        <div class="foot">
          <span class=${"badge " + badgeCls}>${icon} ${label}</span>
          ${data.hasChildren ? html`<span class="tog" title="collapse/expand downstream"
              onClick=${(ev) => { ev.stopPropagation(); data.onToggle(n.id); }}>${data.collapsed ? "+" + data.hiddenCount : "−"}</span>` : null}
        </div>
        <${RF.Handle} type="source" position=${RF.Position.Bottom} style=${{ opacity: 0 }} />
      </div>`;
  }
  const nodeTypes = { r: RNode };

  // ---------------------------------------------------------------- trend chart
  function Trend({ trend, onPick }) {
    const [hover, setHover] = useState(null);
    const [table, setTable] = useState(false);
    if (!trend || !trend.length) return html`<p class="muted">No scored experiments yet.</p>`;
    const W = 360, H = 200, L = 44, R = 10, T = 12, B = 28;
    const vals = trend.flatMap((t) => [t.val_score, t.test_score]).filter((v) => v !== null);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.12 || 0.01; lo -= pad; hi += pad;
    const x = (i) => L + (trend.length === 1 ? (W - L - R) / 2 : (i * (W - L - R)) / (trend.length - 1));
    const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
    let best = -Infinity;
    const bestPts = trend.map((t, i) => { best = Math.max(best, t.val_score); return [x(i), y(best)]; });
    const base = trend.find((t) => t.is_baseline);
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => lo + f * (hi - lo));
    const step = bestPts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `H${p[0]}V${p[1]}`)).join("");
    if (table) {
      return html`<div class="trend"><div class="row"><button onClick=${() => setTable(false)}>Chart</button></div>
        <table class="t"><thead><tr><th>#</th><th>val</th><th>test</th><th>beat threshold</th></tr></thead><tbody>
        ${trend.map((t) => html`<tr key=${t.id}><td>${t.is_baseline ? "baseline" : t.id}</td><td>${fmt(t.val_score)}</td>
          <td>${fmt(t.test_score)}</td><td>${t.is_baseline ? "–" : t.success ? "yes" : "no"}</td></tr>`)}</tbody></table></div>`;
    }
    return html`<div class="trend" style=${{ position: "relative" }}>
      <div class="row" style=${{ alignItems: "center" }}>
        <span class="chip"><span class="sw" style=${{ background: "var(--series-1)" }}></span>validation</span>
        <span class="chip"><span class="sw" style=${{ background: "transparent", border: "2px solid var(--series-2)" }}></span>held-out test</span>
        <span class="chip muted">— best so far · - - baseline</span>
        <button onClick=${() => setTable(true)} style=${{ marginLeft: "auto" }}>Table</button></div>
      <svg viewBox=${`0 0 ${W} ${H}`} role="img" aria-label="Validation and test score per experiment in completion order">
        ${ticks.map((v) => html`<g key=${v}><line x1=${L} x2=${W - R} y1=${y(v)} y2=${y(v)} stroke="var(--grid)" />
          <text x=${L - 5} y=${y(v) + 3} text-anchor="end" font-size="9.5" fill="var(--muted)">${v.toFixed(3)}</text></g>`)}
        <line x1=${L} x2=${W - R} y1=${H - B} y2=${H - B} stroke="var(--axis)" />
        <text x=${(L + W - R) / 2} y=${H - 6} text-anchor="middle" font-size="9.5" fill="var(--muted)">experiments in completion order</text>
        ${base ? html`<line x1=${L} x2=${W - R} y1=${y(base.val_score)} y2=${y(base.val_score)} stroke="var(--muted)" stroke-dasharray="4 3" />` : null}
        <path d=${step} fill="none" stroke="var(--series-1)" stroke-width="2" opacity="0.55" />
        ${trend.map((t, i) => html`<g key=${t.id}>
          ${t.test_score !== null ? html`<circle cx=${x(i)} cy=${y(t.test_score)} r="4" fill="var(--surface)" stroke="var(--series-2)" stroke-width="2" />` : null}
          <circle cx=${x(i)} cy=${y(t.val_score)} r="4.5" fill="var(--series-1)" stroke="var(--surface)" stroke-width="2" />
          <rect x=${x(i) - 10} y=${T} width="20" height=${H - T - B} fill="transparent" style=${{ cursor: "pointer" }}
            onMouseEnter=${() => setHover({ t, i })} onMouseLeave=${() => setHover(null)} onClick=${() => onPick(t.id)} />
        </g>`)}
      </svg>
      ${hover ? html`<div class="tip" style=${{ left: `${(x(hover.i) / W) * 100}%`, top: 0 }}>
        <b>${hover.t.is_baseline ? "Baseline" : "#" + hover.t.id}</b><br/>val ${fmt(hover.t.val_score)} · test ${fmt(hover.t.test_score)}</div>` : null}
    </div>`;
  }

  // ---------------------------------------------------------------- side panel
  function Details({ ctx, detail, graph, onSelect, onTrace, onHighlight, onCompare, compare }) {
    if (!detail) return html`<p class="muted">Click a node to see its record, evidence and lineage.</p>`;
    const n = detail.node, d = n.details || {};
    const byId = graph.byId;
    const link = (id) => byId[id] ? html`<a href="#" onClick=${(e) => { e.preventDefault(); onSelect(id); }}>${byId[id].title}</a>` : id;
    const [cls, icon, label] = st(n.status);
    const evidenceIds = new Set([...(d.new_evidence || []), ...detail.edges.flatMap((e) => e.evidence || [])]);
    const decision = n.type === "DECISION" && html`
      <h3>Why the reasoning changed</h3>
      <div class="kv">
        <div>Before</div><div>${d.before}</div>
        <div>New evidence</div><div>${(d.new_evidence || []).map((id) => html`<div key=${id}>${link(id)}</div>`)}</div>
        <div>Affected</div><div>${(d.affected || []).map((id) => html`<div key=${id}>${link(id)}</div>`)}</div>
        <div>Decision</div><div><b>${label}</b> (${n.epistemic}): ${d.why}</div>
        <div>Alternatives</div><div>${(d.alternatives || []).length ? d.alternatives.map((a) => html`<div key=${a.hypothesis}>${link(a.hypothesis)}</div>`) : html`<span class="muted">none recorded in this round</span>`}</div>
        <div>Next direction</div><div>${(d.why_next || []).length ? d.why_next.map((w) => html`<div key=${w.hypothesis}>${link(w.hypothesis)}<div class="muted">“${w.rationale}”</div></div>`) : html`<span class="muted">no later hypothesis cites this result</span>`}</div>
        <div>Uncertain</div><div>${(d.uncertain || []).length ? d.uncertain.join(" ") : html`<span class="muted">nothing flagged</span>`}</div>
      </div>
      <p class="muted">${d.provenance}</p>`;
    const plain = Object.entries(d).filter(([k]) => !["before", "alternatives", "new_evidence", "affected", "why", "why_next",
      "uncertain", "provenance", "code_versions"].includes(k));
    return html`<div>
      <div class="muted">${n.type.replace("_", " ")} · ${n.origin} · ${when(n.created_at)}</div>
      <h2>${n.title}</h2>
      <div class="row"><span class=${"badge " + (["good", "critical", "warning"].includes(cls) ? cls : "")}>${icon} ${label}</span>
        ${n.confidence ? html`<span class="chip" style=${{ background: "var(--surface-2)" }}>evidence: ${n.confidence}</span>` : null}
        ${n.epistemic ? html`<span class="chip" style=${{ background: "var(--surface-2)" }}>${n.epistemic}</span>` : null}</div>
      ${n.summary ? html`<p>${n.summary}</p>` : null}
      <div class="row">
        <button onClick=${() => onTrace(n.id, "back")}>Trace back to question</button>
        <button onClick=${() => onTrace(n.id, "fwd")}>Trace forward</button>
        ${evidenceIds.size ? html`<button onClick=${() => onHighlight([...evidenceIds], n.id)}>Highlight evidence</button>` : null}
        <button onClick=${() => onCompare(n.id)}>${compare.length === 1 && compare[0] !== n.id ? "Compare with A" : "Compare: set A"}</button>
      </div>
      ${decision}
      ${detail.path_to_question.length ? html`<h3>Path from the question</h3><ol>${detail.path_to_question.map((id) => html`<li key=${id}>${link(id)}</li>`)}</ol>` : null}
      <h3>Relationships</h3>
      ${detail.edges.map((e) => {
        const other = e.src === n.id ? e.dst : e.src;
        const phrase = e.src === n.id ? `${e.rel.replace("_", " ")} →` : `← ${e.rel.replace("_", " ")}`;
        return html`<div class="edge-item" key=${e.id} onClick=${() => onSelect(other)}>
          <span class="rel">${phrase}</span> ${byId[other] ? byId[other].title : other}
          <div class="muted">${e.explanation}${e.created_at ? " · " + when(e.created_at) : ""}</div></div>`;
      })}
      ${plain.length ? html`<h3>Record</h3><pre>${JSON.stringify(Object.fromEntries(plain), null, 2)}</pre>` : null}
      ${detail.code_versions ? html`<h3>Code versions</h3>${detail.code_versions.map((c) => html`<details key=${c.id}>
          <summary>attempt ${c.attempt} · sha ${c.code_sha256.slice(0, 12)}${c.parent_version_id ? " · repairs v" + c.parent_version_id : ""}</summary>
          <p class="muted">${c.explanation}</p><pre>${c.code}</pre></details>`)}` : null}
      ${detail.executions ? html`<h3>Execution logs</h3>${detail.executions.map((x) => html`<details key=${x.attempt}>
          <summary>attempt ${x.attempt} · exit ${x.exit_code === null ? "–" : x.exit_code}${x.timed_out ? " · timed out" : ""} · ${fmt(x.duration_s, 1)}s</summary>
          <pre>${(x.stdout_tail || "") + (x.stderr_tail ? "\n--- stderr ---\n" + x.stderr_tail : "")}</pre></details>`)}` : null}
      ${detail.llm_calls && detail.llm_calls.length ? html`<h3>LLM calls</h3><table class="t"><tbody>${detail.llm_calls.map((c) => html`<tr key=${c.id}>
          <td>${c.role}</td><td>${c.model}</td><td>${c.input_tokens}/${c.output_tokens}</td><td>${c.error ? "⚠ " + c.error.slice(0, 60) : ""}</td></tr>`)}</tbody></table>` : null}
    </div>`;
  }

  function Compare({ data, onSelect, onClose }) {
    const col = (side, name) => html`<div class="col"><b>${name}: ${side.root.title}</b>
      <div class="muted">${st(side.root.status)[2]}</div>
      <h3>Experiments</h3>${side.experiments.length ? side.experiments.map((e) => html`<div key=${e.id}><a href="#" onClick=${(ev) => { ev.preventDefault(); onSelect(e.id); }}>${e.title}</a> <span class="muted">${e.summary}</span></div>`) : html`<div class="muted">none</div>`}
      <h3>Evidence</h3>${side.evidence.map((e) => html`<div key=${e.id}>${st(e.status)[1]} ${e.title} ${e.confidence ? html`<span class="muted">(${e.confidence})</span>` : null}</div>`)}
      <h3>Decisions</h3>${side.decisions.map((e) => html`<div key=${e.id}>${st(e.status)[1]} ${e.title}</div>`)}
      <h3>Led to</h3>${side.hypotheses.length ? side.hypotheses.map((e) => html`<div key=${e.id}>${e.title}</div>`) : html`<div class="muted">nothing yet</div>`}
      ${side.conclusions.length ? html`<h3>Conclusions</h3>${side.conclusions.map((e) => html`<div key=${e.id}>${e.title} (${e.status})</div>`)}` : null}</div>`;
    return html`<div><div class="row"><b>Comparing two lines of research</b><button onClick=${onClose} style=${{ marginLeft: "auto" }}>Close</button></div>
      <div class="compare">${col(data.a, "A")}${col(data.b, "B")}</div>
      <p class="muted">${data.shared.length} downstream node(s) shared by both lines.</p></div>`;
  }

  // ---------------------------------------------------------------- run page (agent trace + checks)
  const PHASE_LABEL = { session: "session", propose: "propose", implement: "implement", execute: "execute",
    score: "score", reflect: "reflect", decide: "decide", stop: "stop" };
  const PHASE_ROLE = { propose: "researcher", implement: "coder", reflect: "analyst",
    execute: "sandbox", score: "trusted scorer", decide: "evaluator rule", session: "controller", stop: "controller" };
  const STEP_ICON = { ok: ["good", "✓"], failed: ["critical", "✕"], running: ["warning", "◔"], interrupted: ["", "⏸"], skipped: ["", "–"] };
  const CHECK_ICON = { ok: ["good", "✓"], warn: ["warning", "!"], fail: ["critical", "✕"], info: ["", "i"] };
  const dur = (s) => (s.started_at && s.finished_at ? `${(s.finished_at - s.started_at).toFixed(1)}s`
    : s.status === "running" && s.started_at ? `${(Date.now() / 1000 - s.started_at).toFixed(0)}s…` : "");
  const clock = (t) => (t ? new Date(t * 1000).toLocaleTimeString() : "");

  // Line diff (LCS). Programs here are a few hundred lines, so O(n*m) is fine.
  function lineDiff(a, b) {
    const A = a.split("\n"), B = b.split("\n"), n = A.length, m = B.length;
    const L = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
    for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
      L[i][j] = A[i] === B[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
    const out = []; let i = 0, j = 0;
    while (i < n && j < m) {
      if (A[i] === B[j]) { out.push([" ", A[i]]); i++; j++; }
      else if (L[i + 1][j] >= L[i][j + 1]) out.push(["-", A[i++]]);
      else out.push(["+", B[j++]]);
    }
    while (i < n) out.push(["-", A[i++]]);
    while (j < m) out.push(["+", B[j++]]);
    return out;
  }

  function Code({ code, previous }) {
    const [mode, setMode] = useState(previous ? "diff" : "full");
    const rows = useMemo(() => (previous && mode === "diff" ? lineDiff(previous.code, code.code) : null), [code, previous, mode]);
    return html`<div>
      <div class="row" style=${{ alignItems: "center" }}>
        <span class="muted">attempt ${code.attempt} · sha ${code.code_sha256.slice(0, 12)}${previous ? ` · repairs attempt ${previous.attempt}` : ""}</span>
        ${previous ? html`<span style=${{ marginLeft: "auto" }}><button class=${mode === "diff" ? "on" : ""} onClick=${() => setMode("diff")}>Diff</button>
          <button class=${mode === "full" ? "on" : ""} onClick=${() => setMode("full")}>Full</button></span>` : null}
      </div>
      ${code.explanation ? html`<p class="muted">${code.explanation}</p>` : null}
      ${rows ? html`<pre class="diff">${rows.filter((r, k) => r[0] !== " " || rows.slice(Math.max(0, k - 2), k + 3).some((x) => x[0] !== " "))
          .map(([t, line], k) => html`<div key=${k} class=${t === "+" ? "add" : t === "-" ? "del" : ""}>${t} ${line}</div>`)}</pre>`
        : html`<pre>${code.code}</pre>`}
    </div>`;
  }

  function StepDetail({ sel, step, onOpenGraph }) {
    const [d, setD] = useState(null);
    useEffect(() => { setD(null); if (step) api("step", { ...sel, id: step.id }).then(setD).catch(() => {}); }, [step && step.id, step && step.status]);
    if (!step) return html`<p class="muted">Select a step to see exactly what the agent was given, what it produced, and what ran.</p>`;
    const [cls, icon] = STEP_ICON[step.status] || ["", "●"];
    const pretty = (txt) => { try { return JSON.stringify(JSON.parse(txt), null, 2); } catch (e) { return txt; } };
    return html`<div>
      <div class="muted">${PHASE_ROLE[step.phase] || ""} · ${step.origin}${step.worker !== null && step.worker !== undefined ? " · worker " + step.worker : ""} · ${when(step.finished_at || step.started_at)}</div>
      <h2>${PHASE_LABEL[step.phase] || step.phase}${step.experiment_id ? ` · experiment #${step.experiment_id}` : ""}${step.attempt ? ` · attempt ${step.attempt}` : ""}</h2>
      <div class="row"><span class=${"badge " + cls}>${icon} ${step.status}</span><span class="muted">${dur(step)}</span>
        ${step.experiment_id ? html`<button style=${{ marginLeft: "auto" }} onClick=${() => onOpenGraph(step.experiment_id)}>Open in graph</button>` : null}</div>
      ${step.hypothesis_title ? html`<p><b>Hypothesis:</b> ${step.hypothesis_title}</p>` : null}
      ${step.summary ? html`<p>${step.summary}</p>` : null}
      ${!d ? html`<p class="muted">Loading…</p>` : html`<div>
        ${d.hypotheses.length ? html`<h3>Hypotheses proposed</h3>${d.hypotheses.map((h) => html`<details key=${h.id}><summary><b>${h.title}</b> <span class="muted">priority ${h.priority} · predicted ${h.expected_metric}</span></summary>
            <p>${h.rationale}</p><p class="muted">${h.method}</p></details>`)}` : null}
        ${d.code ? html`<h3>Program</h3><${Code} code=${d.code} previous=${d.previous_code} />` : null}
        ${d.execution ? html`<h3>Execution</h3><div class="kv">
            <div>exit code</div><div>${d.execution.exit_code === null ? "–" : d.execution.exit_code}${d.execution.timed_out ? " (timed out)" : ""}</div>
            <div>duration</div><div>${fmt(d.execution.duration_s, 1)}s</div>
            <div>image</div><div>${(d.execution.image_id || "").slice(0, 19)}</div></div>
            <details><summary>docker command</summary><pre>${(JSON.parse(d.execution.command || "[]") || []).join(" ")}</pre></details>
            <h3>stdout</h3><pre>${d.execution.stdout_tail || "(empty)"}</pre>
            ${d.execution.stderr_tail ? html`<h3>stderr</h3><pre>${d.execution.stderr_tail}</pre>` : null}` : null}
        ${d.llm_calls.length ? html`<h3>LLM calls (${d.llm_calls.length})</h3>${d.llm_calls.map((c) => html`<div key=${c.id} class="call">
            <div class="row" style=${{ alignItems: "center" }}><b>${c.role}</b><span class="muted">${c.model} · ${c.input_tokens}/${c.output_tokens} tokens · $${fmt(c.cost_usd, 4)}</span>
              ${c.error ? html`<span class="badge critical">✕ ${c.error.slice(0, 80)}</span>` : null}</div>
            <details><summary>system prompt</summary><pre>${c.system}</pre></details>
            <details><summary>prompt</summary><pre>${c.prompt}</pre></details>
            <details open=${!c.error}><summary>response</summary><pre>${c.response ? pretty(c.response) : "(none)"}</pre></details></div>`)}` : null}
        ${Object.keys(step.detail || {}).length ? html`<details><summary>step record</summary><pre>${JSON.stringify(step.detail, null, 2)}</pre></details>` : null}
      </div>`}
    </div>`;
  }

  function RunView({ sel, onOpenGraph }) {
    const [tl, setTl] = useState(null);
    const [checks, setChecks] = useState([]);
    const [picked, setPicked] = useState(null);
    const [phase, setPhase] = useState("");
    const [exp, setExp] = useState("");
    const [problems, setProblems] = useState(false);
    const [err, setErr] = useState(null);
    const load = useCallback(() => {
      if (!sel) return;
      api("steps", sel).then((t) => { setTl(t); setErr(null); }).catch((e) => setErr(e.message));
      api("checks", sel).then(setChecks).catch(() => {});
    }, [sel]);
    useEffect(() => { setTl(null); setPicked(null); load(); }, [load]);
    const live = tl && tl.state.label === "running";
    useEffect(() => { if (!live) return; const t = setInterval(load, 3000); return () => clearInterval(t); }, [live, load]);
    if (err) return html`<div class="err" style=${{ gridRow: "2 / 4" }}>${err}</div>`;
    if (!tl) return html`<div class="muted" style=${{ gridRow: "2 / 4", padding: "12px" }}>Loading…</div>`;
    const steps = tl.steps.filter((s) => (!phase || s.phase === phase) && (!exp || String(s.experiment_id) === exp)
      && (!problems || ["failed", "interrupted"].includes(s.status)));
    const exps = [...new Set(tl.steps.map((s) => s.experiment_id).filter((x) => x))].sort((a, b) => a - b);
    const step = picked && tl.steps.find((s) => s.id === picked);
    const s0 = tl.state;
    const stateBadge = s0.label === "running" ? html`<span class="badge warning"><span class="pulse"></span> running · ${s0.running_steps} step(s) in progress</span>`
      : html`<span class="badge">${s0.label}${s0.stop_reason ? " · " + s0.stop_reason : ""}</span>`;
    const counts = { failed: tl.steps.filter((s) => s.status === "failed").length };
    return html`<div class="runview">
      <div class="runbar">
        ${stateBadge}
        <span class="muted">last activity ${s0.last_activity ? new Date(s0.last_activity * 1000).toLocaleString() : "–"}
          ${tl.source === "reconstructed" ? " · trace reconstructed from records (run predates live step logging)" : ""}</span>
        <span class="grow"></span>
        <span class="lbl">phase</span><select value=${phase} onChange=${(e) => setPhase(e.target.value)}><option value="">all</option>
          ${Object.keys(PHASE_LABEL).map((p) => html`<option key=${p} value=${p}>${p}</option>`)}</select>
        <span class="lbl">experiment</span><select value=${exp} onChange=${(e) => setExp(e.target.value)}><option value="">all</option>
          ${exps.map((x) => html`<option key=${x} value=${String(x)}>#${x}</option>`)}</select>
        <button class=${problems ? "on" : ""} onClick=${() => setProblems(!problems)}>Problems only (${counts.failed})</button>
        <button onClick=${load}>${live ? "Live ●" : "Refresh"}</button>
      </div>
      <div class="runmain">
        <div class="timeline">
          ${steps.map((s) => { const [cls, icon] = STEP_ICON[s.status] || ["", "●"]; return html`
            <div key=${s.id} class=${"tstep " + (picked === s.id ? "sel " : "") + "ph-" + s.phase} onClick=${() => setPicked(s.id)}>
              <span class=${"badge " + cls} title=${s.status}>${icon}</span>
              <span class="tphase">${s.phase}</span>
              <span class="texp">${s.experiment_id ? "#" + s.experiment_id + (s.attempt && s.phase !== "decide" ? "·" + s.attempt : "") : ""}</span>
              <span class="tsum">${s.summary || s.hypothesis_title || ""}</span>
              <span class="ttime">${dur(s)} ${clock(s.finished_at || s.started_at)}</span>
            </div>`; })}
          ${steps.length ? null : html`<p class="muted" style=${{ padding: "12px" }}>No steps match.</p>`}
        </div>
        <div class="side">
          <h3 style=${{ marginTop: 0 }}>Checks</h3>
          ${checks.map((c) => { const [cls, icon] = CHECK_ICON[c.status] || ["", "·"]; return html`<div key=${c.id} class="check">
            <span class=${"badge " + cls}>${icon}</span><div><b>${c.label}</b><div class="muted">${c.detail}</div></div></div>`; })}
          <h3>Step</h3>
          <${StepDetail} sel=${sel} step=${step} onOpenGraph=${onOpenGraph} />
        </div>
      </div>
    </div>`;
  }

  // ---------------------------------------------------------------- app
  function App() {
    const [runs, setRuns] = useState([]);
    const [sel, setSel] = useState(null);           // {db, run}
    const [graph, setGraph] = useState(null);
    const [err, setErr] = useState(null);
    const [hiddenTypes, setHiddenTypes] = useState(DEFAULT_HIDDEN);
    const [statusFilter, setStatusFilter] = useState("");
    const [confFilter, setConfFilter] = useState("");
    const [from, setFrom] = useState(""), [to, setTo] = useState("");
    const [search, setSearch] = useState("");
    const [collapsed, setCollapsed] = useState(new Set());
    const [selected, setSelected] = useState(null);
    const [detail, setDetail] = useState(null);
    const [highlight, setHighlight] = useState(null);  // {nodes:Set, edges:Set, label}
    const [compare, setCompare] = useState([]);
    const [compareData, setCompareData] = useState(null);
    const [tab, setTab] = useState("details");
    const [mode, setMode] = useState(new URLSearchParams(location.search).get("view") === "graph" ? "graph" : "run");
    const rf = useRef(null);

    useEffect(() => {
      api("runs", {}).then((rs) => {
        setRuns(rs);
        const p = new URLSearchParams(location.search);
        const pick = rs.find((r) => r.db === p.get("db") && String(r.id) === p.get("run")) || rs[rs.length - 1];
        if (pick) setSel({ db: pick.db, run: pick.id });
      }).catch((e) => setErr(e.message));
    }, []);

    const load = useCallback(() => {
      if (!sel) return;
      api("graph", sel).then((g) => {
        g.byId = Object.fromEntries(g.nodes.map((n) => [n.id, n]));
        setGraph(g); setErr(null);
      }).catch((e) => setErr(e.message));
    }, [sel]);
    useEffect(() => { setSelected(null); setDetail(null); setHighlight(null); setCompare([]); setCompareData(null); load(); }, [load]);

    const select = useCallback((id) => {
      setSelected(id); setTab("details");
      setHighlight((h) => (h && !h.nodes.has(id) ? null : h));
      api("node", { ...sel, id }).then(setDetail).catch((e) => setErr(e.message));
      const inst = rf.current;
      if (inst && inst.getNode(id)) inst.fitView({ nodes: [{ id }], duration: 400, maxZoom: 1.1 });
    }, [sel]);

    // Visible set: filters, then collapse (hide everything only reachable through a collapsed node).
    const view = useMemo(() => {
      if (!graph) return null;
      const t0 = from ? new Date(from).getTime() / 1000 : -Infinity;
      const t1 = to ? new Date(to).getTime() / 1000 + 86400 : Infinity;
      const keep = graph.nodes.filter((n) => !hiddenTypes.has(n.type)
        && (!statusFilter || n.status === statusFilter)
        && (!confFilter || (confFilter === "strong" ? n.confidence === "strong" : ["strong", "moderate"].includes(n.confidence)))
        && (n.created_at || 0) >= t0 && (n.created_at || 0) <= t1);
      const ids = new Set(keep.map((n) => n.id));
      // Hidden *types* are bridged (question -> [method] -> experiment stays connected);
      // nodes removed by status/date/confidence filters are not.
      const outAll = {};
      graph.edges.forEach((e) => { (outAll[e.flow_src] = outAll[e.flow_src] || []).push(e); });
      const byId = graph.byId, edges = [], seenPair = new Set();
      keep.forEach((u) => (outAll[u.id] || []).forEach((e0) => {
        if (ids.has(e0.flow_dst)) { edges.push(e0); return; }
        const stack = [[e0.flow_dst, [byId[e0.flow_dst] && byId[e0.flow_dst].type]]], visited = new Set();
        while (stack.length) {
          const [x, via] = stack.pop();
          if (visited.has(x) || !byId[x] || !hiddenTypes.has(byId[x].type)) continue;
          visited.add(x);
          (outAll[x] || []).forEach((e) => {
            if (ids.has(e.flow_dst)) {
              const key = u.id + ">" + e.flow_dst;
              if (!seenPair.has(key)) { seenPair.add(key);
                edges.push({ id: "via:" + key, src: u.id, dst: e.flow_dst, flow_src: u.id, flow_dst: e.flow_dst,
                  rel: "via", via: [...new Set(via)], explanation: "Connected through hidden " + [...new Set(via)].join(", ").toLowerCase() + " node(s)", evidence: [] }); }
            } else stack.push([e.flow_dst, [...via, byId[e.flow_dst] && byId[e.flow_dst].type]]);
          });
        }
      }));
      const down = {}, indeg = {};
      edges.forEach((e) => { (down[e.flow_src] = down[e.flow_src] || []).push(e.flow_dst); indeg[e.flow_dst] = (indeg[e.flow_dst] || 0) + 1; });
      const roots = keep.filter((n) => !indeg[n.id]).map((n) => n.id);
      const seen = new Set(roots), stack = [...roots];
      while (stack.length) {
        const c = stack.pop();
        if (collapsed.has(c)) continue;
        (down[c] || []).forEach((d) => { if (!seen.has(d)) { seen.add(d); stack.push(d); } });
      }
      const hiddenUnder = {};
      collapsed.forEach((c) => {
        const s = new Set(), st2 = [...(down[c] || [])];
        while (st2.length) { const x = st2.pop(); if (!s.has(x) && !seen.has(x)) { s.add(x); (down[x] || []).forEach((y) => st2.push(y)); } }
        hiddenUnder[c] = s.size;
      });
      const nodes = keep.filter((n) => seen.has(n.id));
      const vEdges = edges.filter((e) => seen.has(e.flow_src) && seen.has(e.flow_dst));
      return { nodes, edges: vEdges, down, hiddenUnder };
    }, [graph, hiddenTypes, statusFilter, confFilter, from, to, collapsed]);

    const q = search.trim().toLowerCase();
    const matches = useMemo(() => new Set(!q || !view ? [] : view.nodes.filter((n) =>
      (n.title + " " + (n.summary || "") + " " + JSON.stringify(n.details || {})).toLowerCase().includes(q)).map((n) => n.id)), [q, view]);

    const toggle = useCallback((id) => setCollapsed((c) => { const s = new Set(c); s.has(id) ? s.delete(id) : s.add(id); return s; }), []);

    const rfNodes = useMemo(() => {
      if (!view) return [];
      const pos = layout(view.nodes, view.edges);
      return view.nodes.map((n) => ({
        id: n.id, type: "r", position: pos[n.id],
        data: { node: n, match: matches.has(n.id), selected: selected === n.id, onToggle: toggle,
          hasChildren: (view.down[n.id] || []).length > 0 || collapsed.has(n.id), collapsed: collapsed.has(n.id),
          hiddenCount: view.hiddenUnder[n.id] || 0,
          dim: highlight ? !highlight.nodes.has(n.id) : false },
      }));
    }, [view, matches, selected, highlight, toggle, collapsed]);

    const rfEdges = useMemo(() => (view ? view.edges.map((e) => {
      const lit = highlight && highlight.edges.has(e.id);
      const style = { ...(EDGE_STYLE[e.rel] || { stroke: "var(--axis)" }), ...(highlight && !lit ? { opacity: 0.12 } : {}),
        ...(lit ? { strokeWidth: 3 } : {}) };
      return { id: e.id, source: e.flow_src, target: e.flow_dst, label: edgeLabel(e), style,
        markerEnd: { type: RF.MarkerType.ArrowClosed, width: 14, height: 14 }, data: e };
    }) : []), [view, highlight]);

    useEffect(() => {
      if (q && matches.size && rf.current) rf.current.fitView({ nodes: [...matches].map((id) => ({ id })), duration: 400, maxZoom: 1 });
    }, [q, matches]);

    const trace = (id, dir) => api("trace", { ...sel, id, dir }).then((r) => {
      setHighlight({ nodes: new Set(r.nodes), edges: new Set(r.edges), label: dir === "back" ? "Traced back to the question" : "Everything downstream" });
      if (rf.current) rf.current.fitView({ nodes: r.nodes.map((x) => ({ id: x })), duration: 400 });
    });
    const highlightEvidence = (ids, from) => {
      const nodes = new Set([from, ...ids]);
      const edges = new Set(graph.edges.filter((e) => (nodes.has(e.src) && nodes.has(e.dst))).map((e) => e.id));
      setHighlight({ nodes, edges, label: "Evidence behind this node" });
    };
    const unexplored = () => {
      const nodes = new Set(graph.nodes.filter((n) => ["not_investigated", "untested_invalid", "open", "unverified", "questioned", "not_supported"].includes(n.status)).map((n) => n.id));
      setHighlight({ nodes, edges: new Set(), label: "Unexplored, untested or unresolved" });
    };
    const pickCompare = (id) => {
      if (compare.length === 1 && compare[0] !== id) {
        api("compare", { ...sel, a: compare[0], b: id }).then((d) => { setCompareData(d); setTab("compare"); });
        setCompare([]);
      } else setCompare([id]);
    };
    const openGraph = (expId) => { setMode("graph"); setTimeout(() => select(`r${sel.run}:exp:${expId}`), 50); };
    const pickExperiment = (expId) => { const n = graph.nodes.find((x) => x.type === "EXPERIMENT" && x.source_id === expId); if (n) select(n.id); };

    const statuses = graph ? [...new Set(graph.nodes.map((n) => n.status))].sort() : [];
    const qs = sel ? new URLSearchParams(sel).toString() : "";

    return html`<div class="app">
      <div class="top">
        <h1>Research</h1>
        <span class="seg"><button class=${mode === "run" ? "on" : ""} onClick=${() => setMode("run")}>Run</button><button
          class=${mode === "graph" ? "on" : ""} onClick=${() => setMode("graph")}>Graph</button></span>
        <select value=${sel ? sel.db + "|" + sel.run : ""} onChange=${(e) => { const [db, run] = e.target.value.split("|"); setSel({ db, run: Number(run) }); }}>
          ${runs.map((r) => html`<option key=${r.db + r.id} value=${r.db + "|" + r.id}>${r.name} (${r.domain}, ${r.experiments} exp)</option>`)}
        </select>
        ${mode === "graph" ? html`
        <input type="text" placeholder="Search titles, rationale, results…" value=${search} onInput=${(e) => setSearch(e.target.value)} style=${{ minWidth: "220px" }} />
        <span class="muted">${q ? matches.size + " match(es)" : ""}</span>
        <span class="grow"></span>
        <button onClick=${unexplored}>Show unexplored</button>
        ${highlight ? html`<button class="on" onClick=${() => setHighlight(null)}>✕ ${highlight.label}</button>` : null}
        <button onClick=${load}>Refresh</button>` : html`<span class="grow"></span>`}
        <a href=${"/api/export.json?" + qs}>Export JSON</a>
        <a href=${"/api/export.md?" + qs}>Export Markdown</a>
      </div>
      ${mode === "run" ? html`<${RunView} sel=${sel} onOpenGraph=${openGraph} />` : html`<${React.Fragment}>
      <div class="filters">
        ${TYPES.map((t) => html`<button key=${t} class=${"chip " + (hiddenTypes.has(t) ? "" : "on")}
            onClick=${() => setHiddenTypes((h) => { const s = new Set(h); s.has(t) ? s.delete(t) : s.add(t); return s; })}>
            <span class="sw" style=${{ background: `var(--t-${t.toLowerCase()})` }}></span>${t.replace("_", " ").toLowerCase()}</button>`)}
        <span class="lbl">status</span>
        <select value=${statusFilter} onChange=${(e) => setStatusFilter(e.target.value)}><option value="">any</option>
          ${statuses.map((s) => html`<option key=${s} value=${s}>${st(s)[2]}</option>`)}</select>
        <span class="lbl">evidence</span>
        <select value=${confFilter} onChange=${(e) => setConfFilter(e.target.value)}><option value="">any</option>
          <option value="moderate">moderate or strong</option><option value="strong">strong</option></select>
        <span class="lbl">from</span><input type="date" value=${from} onChange=${(e) => setFrom(e.target.value)} />
        <span class="lbl">to</span><input type="date" value=${to} onChange=${(e) => setTo(e.target.value)} />
      </div>
      <div class="main">
        <div class="canvas">
          ${err ? html`<div class="err">${err}</div>` : null}
          <${RF.default} nodes=${rfNodes} edges=${rfEdges} nodeTypes=${nodeTypes} fitView minZoom=${0.05}
            nodesDraggable=${false} nodesConnectable=${false} onInit=${(i) => { rf.current = i; }}
            onNodeClick=${(_, n) => select(n.id)} onPaneClick=${() => setHighlight(null)} proOptions=${{ hideAttribution: true }}>
            <${RF.Background} gap=${24} color="var(--grid)" />
            <${RF.Controls} showInteractive=${false} />
            <${RF.MiniMap} pannable zoomable nodeColor=${(n) => getComputedStyle(document.documentElement).getPropertyValue(`--t-${n.data.node.type.toLowerCase()}`)} />
          </${RF.default}>
          <div class="legend">
            <div class="l"><span class="box" style=${{ borderColor: "var(--good)" }}></span>✓ supported / current conclusion</div>
            <div class="l"><span class="box" style=${{ borderColor: "var(--critical)", borderStyle: "dashed" }}></span>✕ contradicted by experiment (empirical)</div>
            <div class="l"><span class="box" style=${{ borderColor: "var(--warning)", borderStyle: "dashed" }}></span>~ not supported / ? open or questioned</div>
            <div class="l"><span class="box" style=${{ opacity: 0.6 }}></span>↓ beat baseline but not best · ⤳ superseded</div>
            <div class="l"><span class="box" style=${{ borderStyle: "dashed" }}></span>⚠ invalid experiment (hypothesis untested)</div>
            <div class="l"><span class="box" style=${{ borderStyle: "dotted" }}></span>○ not investigated / unverified</div>
          </div>
        </div>
        <div class="side">
          <div class="tabs">
            ${["details", "next", "trend", "compare"].map((t) => html`<button key=${t} class=${tab === t ? "on" : ""} onClick=${() => setTab(t)}>
              ${{ details: "Details", next: "Next steps", trend: "Trend", compare: "Compare" }[t]}</button>`)}
          </div>
          ${compare.length === 1 ? html`<p class="muted">A = ${graph && graph.byId[compare[0]] ? graph.byId[compare[0]].title : compare[0]}. Select another node and press “Compare with A”.</p>` : null}
          ${tab === "details" && graph ? html`<${Details} detail=${detail} graph=${graph} onSelect=${select} onTrace=${trace}
              onHighlight=${highlightEvidence} onCompare=${pickCompare} compare=${compare} />` : null}
          ${tab === "next" && graph ? html`<div><h3>What to investigate next</h3>
              ${graph.next_steps.map((s) => html`<div class="next-item" key=${s.node + s.kind} onClick=${() => select(s.node)}>
                <b>${s.title}</b><div class="muted">${s.kind.replace("_", " ")} · ${s.reason}</div></div>`)}
              ${graph.next_steps.length ? null : html`<p class="muted">Nothing open.</p>`}</div>` : null}
          ${tab === "trend" && graph ? html`<div><h3>Performance by experiment</h3><${Trend} trend=${graph.trend} onPick=${pickExperiment} />
              <p class="muted">Agents only ever saw validation scores. Test scores are shown here so you can spot overfitting to validation.</p></div>` : null}
          ${tab === "compare" ? (compareData ? html`<${Compare} data=${compareData} onSelect=${select} onClose=${() => { setCompareData(null); setTab("details"); }} />`
              : html`<p class="muted">Select a node, press “Compare: set A”, then select a second node and press “Compare with A”.</p>`) : null}
        </div>
      </div></${React.Fragment}>`}
    </div>`;
  }

  ReactDOM.createRoot(document.getElementById("root")).render(html`<${RF.ReactFlowProvider}><${App} /></${RF.ReactFlowProvider}>`);
})();
