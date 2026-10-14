"""Research Reasoning Graph: an explicit, persisted graph of how a run's conclusions came about.

Nodes are research artifacts (question, assumptions, hypotheses, methods, experiments,
evidence, decisions, conclusions, open questions); edges are typed relationships with a
timestamp, a plain-language explanation and the evidence node ids behind them.

Everything here is *derived from records the loop already persists* (runs, hypotheses,
experiments, executions, llm_calls). Nothing is inferred from hidden model reasoning: text
fields are either quoted from what an agent wrote (`recorded`) or computed from numbers
(`derived`), and every node says which.

Persistence is additive. Derived nodes get deterministic ids, so rebuilding is an upsert;
a derived node that no longer follows from the records is marked `stale`, never deleted.
Nodes and edges with origin other than `derived` (e.g. added by a human) are never touched.

Epistemic care, which the builder enforces:
- A completed experiment that misses the success threshold *contradicts* its hypothesis
  empirically (one validation split); it is never a proof of impossibility.
- A hypothesis that beat the baseline but not the current best is *deprioritized*, which is
  a benchmark ranking, not a refutation.
- A crashed or invalid experiment is not evidence about the hypothesis; the hypothesis stays
  *untested*.
- `rules_out` and `epistemic="formal"` are never produced automatically.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .memory import Memory

NODE_TYPES = ("QUESTION", "ASSUMPTION", "OBSERVATION", "HYPOTHESIS", "METHOD", "EXPERIMENT",
              "EVIDENCE", "DECISION", "CONCLUSION", "OPEN_QUESTION")

# Every edge reads "src <rel> dst". `flow` orients it from cause to effect in the reasoning
# (question -> hypothesis -> experiment -> evidence -> decision -> conclusion), which is what
# layout, backward/forward tracing and collapsing use.
#   "fwd": src is upstream of dst.  "rev": dst is upstream of src.
#   "auto": upstream is the hypothesis/method end if there is one (evidence supports a
#           hypothesis that came before it), otherwise src (evidence supports a conclusion).
RELATIONS = {
    "motivates": "fwd", "assumes": "fwd", "proposes": "fwd", "tests": "rev",
    "supports": "auto", "contradicts": "auto", "refines": "rev", "supersedes": "rev",
    "depends_on": "rev", "inspired_by": "rev", "rules_out": "rev",
    # Additions beyond the core list, kept separate so weak evidence is never phrased as a proof:
    "produces": "fwd",        # experiment -> its evidence
    "selects": "rev",         # decision -> hypothesis adopted as the current best
    "deprioritizes": "rev",   # decision -> hypothesis set aside on empirical/benchmark grounds
}

VERDICT_LABEL = {"adopt": "Adopt", "deprioritize": "Deprioritize", "reject": "Reject (empirical)",
                 "set_aside": "Set aside", "invalid": "Invalid run"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    id TEXT PRIMARY KEY, run_id INTEGER, type TEXT, title TEXT, summary TEXT, status TEXT,
    confidence TEXT, epistemic TEXT, details_json TEXT, source_table TEXT, source_id INTEGER,
    origin TEXT DEFAULT 'derived', created_at REAL, updated_at REAL
);
CREATE TABLE IF NOT EXISTS graph_edges (
    id TEXT PRIMARY KEY, run_id INTEGER, src TEXT, dst TEXT, rel TEXT, explanation TEXT,
    evidence_json TEXT, origin TEXT DEFAULT 'derived', created_at REAL, updated_at REAL,
    stale INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS graph_nodes_run ON graph_nodes(run_id);
CREATE INDEX IF NOT EXISTS graph_edges_run ON graph_edges(run_id);
"""


@dataclass
class Node:
    id: str
    type: str
    title: str
    summary: str = ""
    status: str = ""
    confidence: str | None = None      # strong | moderate | weak | None (not assessable)
    epistemic: str | None = None       # empirical | benchmark | invalid | not_investigated | formal | constraint
    details: dict = field(default_factory=dict)
    source_table: str | None = None
    source_id: int | None = None
    created_at: float | None = None
    origin: str = "derived"


@dataclass
class Edge:
    src: str
    rel: str
    dst: str
    explanation: str = ""
    evidence: list[str] = field(default_factory=list)
    created_at: float | None = None
    origin: str = "derived"

    @property
    def id(self) -> str:
        return f"{self.src}|{self.rel}|{self.dst}"


@dataclass
class Graph:
    run: dict
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)

    def add(self, n: Node) -> Node:
        assert n.type in NODE_TYPES, n.type
        self.nodes[n.id] = n
        return n

    def link(self, src: str, rel: str, dst: str, explanation: str = "", evidence: Iterable[str] = (),
             at: float | None = None) -> Edge:
        assert rel in RELATIONS, rel
        e = Edge(src, rel, dst, explanation, list(evidence), at)
        self.edges[e.id] = e
        return e

    # --- traversal --------------------------------------------------------
    def flow(self, e: Edge) -> tuple[str, str]:
        """(upstream, downstream) for an edge."""
        mode = RELATIONS.get(e.rel, "fwd")
        if mode == "auto":
            dst_type = self.nodes[e.dst].type if e.dst in self.nodes else ""
            mode = "rev" if dst_type in ("HYPOTHESIS", "METHOD") else "fwd"
        return (e.dst, e.src) if mode == "rev" else (e.src, e.dst)

    def _adj(self, direction: str) -> dict[str, list[Edge]]:
        adj: dict[str, list[Edge]] = {}
        for e in self.edges.values():
            up, down = self.flow(e)
            adj.setdefault(down if direction == "back" else up, []).append(e)
        return adj

    def trace(self, start: str, direction: str) -> tuple[set[str], set[str]]:
        """All nodes/edges reachable from `start` going upstream ("back") or downstream ("fwd")."""
        adj, seen, edges, stack = self._adj(direction), {start}, set(), [start]
        while stack:
            cur = stack.pop()
            for e in adj.get(cur, []):
                up, down = self.flow(e)
                nxt = up if direction == "back" else down
                edges.add(e.id)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen, edges

    # Edges that jump straight from a result to a hypothesis or decision; the narrated path
    # prefers the longer evidence <- experiment <- hypothesis chain.
    SHORTCUTS = ("supports", "contradicts", "selects", "deprioritizes", "rules_out", "refines")

    def path_to_question(self, start: str) -> list[str]:
        """Upstream chain of node ids from `start` back to the QUESTION, via experiments where possible."""
        import heapq
        adj, dist, prev, heap = self._adj("back"), {start: 0}, {start: None}, [(0, start)]
        while heap:
            d, cur = heapq.heappop(heap)
            if d > dist[cur]:
                continue
            if self.nodes.get(cur) and self.nodes[cur].type == "QUESTION":
                chain = []
                while cur is not None:
                    chain.append(cur)
                    cur = prev[cur]
                return chain
            for e in adj.get(cur, []):
                up = self.flow(e)[0]
                if self.nodes.get(up) and self.nodes[up].type in ("ASSUMPTION", "OPEN_QUESTION"):
                    continue        # context, not lineage: never part of "how we got here"
                shortcut = e.rel in self.SHORTCUTS and self.nodes.get(up) and self.nodes[up].type != "EVIDENCE"
                # An older conclusion is history, not the route by which this one was reached.
                nd = d + (25 if e.rel == "supersedes" else 3 if shortcut else 1)
                if nd < dist.get(up, float("inf")):
                    dist[up], prev[up] = nd, cur
                    heapq.heappush(heap, (nd, up))
        return []

    def to_dict(self) -> dict:
        out_edges = []
        for e in self.edges.values():
            up, down = self.flow(e)
            out_edges.append({"id": e.id, "src": e.src, "rel": e.rel, "dst": e.dst, "explanation": e.explanation,
                              "evidence": e.evidence, "created_at": e.created_at, "origin": e.origin,
                              "flow_src": up, "flow_dst": down})
        return {
            "run": {k: self.run.get(k) for k in ("id", "name", "domain", "objective", "created_at")},
            "nodes": [{**n.__dict__} for n in self.nodes.values()],
            "edges": out_edges,
        }


# --- statistics ------------------------------------------------------------
def strength(improvement: float, p_base: float, p_new: float, n: int | None) -> tuple[str | None, float | None]:
    """Approximate z-score of a difference of two proportions on one split of size n.

    Unpaired normal approximation: deliberately rough, used only to label evidence
    weak/moderate/strong so a 2-sample difference is not presented as a finding.
    """
    if not n or not (0 <= p_base <= 1 and 0 <= p_new <= 1):
        return None, None
    se = math.sqrt(max(p_base * (1 - p_base) + p_new * (1 - p_new), 1e-12) / n)
    z = abs(improvement) / se
    return ("strong" if z >= 3 else "moderate" if z >= 2 else "weak"), round(z, 2)


def _short(text: str | None, n: int = 90) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    first = re.split(r"(?<=[.;:])\s", text, maxsplit=1)[0]
    s = first if len(first) <= n else first[: n - 1].rstrip() + "..."
    return s


def _fmt(x: float | None, nd: int = 4) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


# --- builder -----------------------------------------------------------------
class Builder:
    """Derives the graph for one run from the loop's existing tables."""

    def __init__(self, memory: Memory, run: dict, *, work_dir: Path | None = None, domain=None):
        self.m, self.run, self.rid = memory, run, run["id"]
        self.work_dir = work_dir
        self.domain = domain
        self.g = Graph(run)
        self._pred_cache: dict[int, list[str] | None] = {}

    def nid(self, kind: str, key) -> str:
        return f"r{self.rid}:{kind}:{key}"

    def n_val(self) -> int | None:
        if not self.work_dir:
            return None
        p = self.work_dir / "data" / "private" / "val_y.csv"
        try:
            return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        except OSError:
            return None

    def build(self) -> Graph:
        g, run = self.g, self.run
        dom = self.domain
        metric = dom.metric if dom else "score"
        higher = dom.higher_is_better if dom else True
        threshold = dom.min_improvement if dom else 0.0
        n = self.n_val()
        exps = self.m.experiments(self.rid)
        hyps = self.m.hypotheses(self.rid)
        base = next((e for e in exps if e["is_baseline"] and e["status"] == "completed"), None)
        better = (lambda a, b: a > b) if higher else (lambda a, b: a < b)

        # Question and the assumptions every conclusion in this run rests on.
        q = g.add(Node(self.nid("question", 0), "QUESTION", run["objective"] or "(no objective recorded)",
                       summary=f"Domain `{run['domain']}`, metric: validation {metric} "
                               f"({'higher' if higher else 'lower'} is better).",
                       status="open", details={"domain": run["domain"], "metric": metric,
                                               "higher_is_better": higher, "min_improvement": threshold,
                                               "n_val": n},
                       source_table="runs", source_id=self.rid, created_at=run["created_at"]))
        a_proxy = g.add(Node(self.nid("assumption", "val_proxy"), "ASSUMPTION",
                             f"Validation {metric} on one fixed split reflects held-out performance",
                             status="unverified", epistemic="empirical", created_at=run["created_at"],
                             details={"why": "Agents only ever see validation scores; every decision is made on them."}))
        a_thresh = g.add(Node(self.nid("assumption", "threshold"), "ASSUMPTION",
                              f"An improvement of at least {threshold} over the baseline is a real gain",
                              status="unverified", epistemic="empirical", created_at=run["created_at"],
                              details={"min_improvement": threshold}))
        sandbox = json.loads(run.get("config_json") or "{}").get("sandbox", {})
        a_env = g.add(Node(self.nid("assumption", "sandbox"), "ASSUMPTION",
                           "Methods are limited to one CPU, no network and the libraries in the sandbox image",
                           status="accepted", epistemic="constraint", created_at=run["created_at"],
                           details={"sandbox": sandbox}))
        for a in (a_proxy, a_thresh, a_env):
            g.link(q.id, "assumes", a.id, "Accepted when the run was configured.", at=run["created_at"])

        # Baseline: the reference method every hypothesis is measured against.
        ev_of: dict[int, str] = {}
        exp_node: dict[int, str] = {}
        if base:
            code = self.m.one("SELECT code FROM code_versions WHERE experiment_id=? ORDER BY attempt LIMIT 1", base["id"])
            m = re.search(r"#\s*Baseline:\s*(.+)", (code or {}).get("code") or "")
            bm = g.add(Node(self.nid("method", "baseline"), "METHOD",
                            "Baseline: " + (m.group(1).strip().rstrip(".") if m else "domain reference program"),
                            status="reference", details={"code_version": base["code_version_id"]},
                            source_table="experiments", source_id=base["id"], created_at=base["started_at"]))
            g.link(q.id, "motivates", bm.id, "Every candidate is compared with this fixed reference.",
                   at=base["started_at"])
            exp_node[base["id"]] = self._experiment(base, None, bm.id)
            ev = g.add(Node(self.nid("evidence", f"e{base['id']}"), "EVIDENCE",
                            f"Baseline val {metric} {_fmt(base['val_score'])}", status="reference",
                            epistemic="empirical", details={"val": base["val_score"], "test": base["test_score"],
                                                             "n_val": n},
                            source_table="experiments", source_id=base["id"], created_at=base["finished_at"]))
            g.link(exp_node[base["id"]], "produces", ev.id, "Scored by the trusted evaluator.", at=base["finished_at"])
            ev_of[base["id"]] = ev.id
        bval = base["val_score"] if base else None

        # Hypotheses, their methods and the experiments that tested them.
        by_hyp: dict[int, list[dict]] = {}
        for e in exps:
            if not e["is_baseline"] and e["hypothesis_id"]:
                by_hyp.setdefault(e["hypothesis_id"], []).append(e)
        round_of: dict[int, list[dict]] = {}
        for h in hyps:
            round_of.setdefault(h["llm_call_id"] or -h["id"], []).append(h)
        hyp_id = {h["id"]: self.nid("hyp", h["id"]) for h in hyps}
        # Which experiment's hypothesis a parent experiment tested, for `refines` edges.
        hyp_of_exp = {e["id"]: e["hypothesis_id"] for e in exps if e["hypothesis_id"]}

        for h in hyps:
            hid = hyp_id[h["id"]]
            parents = json.loads(h["parent_experiment_ids"] or "[]")
            g.add(Node(hid, "HYPOTHESIS", h["title"], summary=_short(h["rationale"], 220),
                       details={"rationale": h["rationale"], "expected_metric": h["expected_metric"],
                                "priority": h["priority"], "parent_experiment_ids": parents,
                                "researcher_call": h["llm_call_id"], "text_origin": "recorded (researcher)"},
                       source_table="hypotheses", source_id=h["id"], created_at=h["created_at"]))
            mid = g.add(Node(self.nid("method", f"h{h['id']}"), "METHOD", f"Method: {h['title']}"[:120],
                             summary=h["method"], details={"method": h["method"], "text_origin": "recorded (researcher)"},
                             source_table="hypotheses", source_id=h["id"], created_at=h["created_at"])).id
            g.link(hid, "proposes", mid, "Method the researcher proposed to test this hypothesis.", at=h["created_at"])
            if not parents:
                g.link(q.id, "motivates", hid, "Proposed without citing an earlier experiment.", at=h["created_at"])
            for p in parents:
                parent_is_base = base is not None and p == base["id"]
                target = self.nid("evidence" if parent_is_base else "decision", f"e{p}")
                g.link(hid, "inspired_by", target,
                       f"The researcher cited experiment #{p} as motivation.", [self.nid("evidence", f"e{p}")],
                       at=h["created_at"])
                ph = hyp_of_exp.get(p)
                if ph and ph != h["id"] and ph in hyp_id:
                    g.link(hid, "refines", hyp_id[ph], f"Builds on the hypothesis tested in experiment #{p}.",
                           at=h["created_at"])
            for e in by_hyp.get(h["id"], []):
                exp_node[e["id"]] = self._experiment(e, hid, mid)

        # Evidence and decisions, in the order results arrived.
        finished = sorted((e for e in exps if not e["is_baseline"] and e["status"] in ("completed", "failed")),
                          key=lambda e: (e["finished_at"] or 0, e["id"]))
        best: dict | None = None
        conclusions: list[str] = []
        outcome_of_hyp: dict[int, list[str]] = {}
        children = {}
        for h in hyps:
            for p in json.loads(h["parent_experiment_ids"] or "[]"):
                children.setdefault(p, []).append(h)
        for e in finished:
            hid = hyp_id.get(e["hypothesis_id"])
            evid = self.nid("evidence", f"e{e['id']}")
            did = self.nid("decision", f"e{e['id']}")
            siblings = [s for s in round_of.get(self._round_key(e, hyps), []) if s["id"] != e["hypothesis_id"]]
            before = (f"Predicted val {metric} {_fmt(e['expected_metric'])}. Best before this result: "
                      + (f"#{best['id']} {_fmt(best['val_score'])}" if best else f"baseline {_fmt(bval)}") + ".")
            why_next = [{"hypothesis": hyp_id[c["id"]], "title": c["title"], "rationale": c["rationale"]}
                        for c in children.get(e["id"], [])]
            alternatives = [{"hypothesis": hyp_id[s["id"]], "title": s["title"]} for s in siblings]
            if e["status"] == "failed":
                g.add(Node(evid, "EVIDENCE", f"Invalid experiment #{e['id']}: no score",
                           summary=_short(e["error"], 300), status="invalid", epistemic="invalid",
                           details={"error": e["error"], "attempts": e["attempts"]},
                           source_table="experiments", source_id=e["id"], created_at=e["finished_at"]))
                g.link(exp_node[e["id"]], "produces", evid, "The program never produced scoreable output.",
                       at=e["finished_at"])
                verdict, rel, why = ("invalid", None,
                                     "The experiment failed, so it says nothing about the hypothesis; "
                                     "the hypothesis remains untested.")
                uncertain = ["Whether the hypothesis holds: it was never measured."]
                outcome_of_hyp.setdefault(e["hypothesis_id"], []).append("invalid")
            else:
                imp, val = e["improvement"], e["val_score"]
                conf, z, zinfo = self.cmp(e, base, imp or 0.0)
                success = bool(e["success"])
                rel_h = "supports" if success else "contradicts"
                g.add(Node(evid, "EVIDENCE", f"val {metric} {_fmt(val)} ({imp:+.4f} vs baseline)",
                           summary=f"Held-out test {_fmt(e['test_score'])} (never shown to agents).",
                           status="supports" if success else "contradicts", confidence=conf, epistemic="empirical",
                           details={"val": val, "test": e["test_score"], "baseline_val": e["baseline_val"],
                                    "improvement": imp, "threshold": threshold, "vs_baseline": zinfo, "n_val": n,
                                    "analyst_verdict": e["verdict"], "analyst_analysis": e["analysis"],
                                    "analyst_lessons": json.loads(e["lessons"] or "[]"),
                                    "note": "Outcome is decided by the numeric threshold; analyst text is commentary."},
                           source_table="experiments", source_id=e["id"], created_at=e["finished_at"]))
                g.link(exp_node[e["id"]], "produces", evid, "Scored by the trusted evaluator.", at=e["finished_at"])
                if hid:
                    g.link(evid, rel_h, hid,
                           (f"Improvement {imp:+.4f} {'meets' if success else 'misses'} the {threshold} threshold "
                            f"on one validation split" + (f" ({z}: {conf} evidence)." if conf else ".")),
                           [evid], at=e["finished_at"])
                new_best = success and (best is None or better(val, best["val_score"]))
                if new_best:
                    verdict, rel = "adopt", "selects"
                    why = (f"Best validation {metric} so far"
                           + (f", ahead of #{best['id']} by {abs(val - best['val_score']):.4f}" if best else "")
                           + "; it becomes the current best method.")
                    uncertain = []
                    if best:
                        c2, z2, _ = self.cmp(e, best, val - best["val_score"])
                        if c2 == "weak":
                            uncertain.append(f"The lead over #{best['id']} is within noise ({z2}).")
                elif success:
                    verdict, rel = "deprioritize", "deprioritizes"
                    rel_to_best = ("ties" if val == best["val_score"] else "trails")
                    why = (f"Beat the baseline but {rel_to_best} the current best #{best['id']} "
                           f"({_fmt(best['val_score'])})"
                           + ("; the earlier result stays current" if rel_to_best == "ties" else "")
                           + ". A benchmark ranking, not a refutation.")
                    c2, z2, _ = self.cmp(e, best, val - best["val_score"])
                    uncertain = ([f"The gap to #{best['id']} is within noise ({z2}); it may be as good."]
                                 if c2 == "weak" else [])
                elif (imp or 0) < 0 and conf in ("moderate", "strong"):
                    verdict, rel = "reject", "deprioritizes"
                    why = (f"Clearly worse than the baseline (improvement {imp:+.4f}, {z}). Empirically "
                           "contradicted on this split; not shown to be impossible.")
                    uncertain = ["A different configuration of the same idea could still help."]
                else:
                    verdict, rel = "set_aside", "deprioritizes"
                    why = ((f"Lower than the baseline ({imp:+.4f}), but not significantly ({z}). " if (imp or 0) < 0
                            else f"No detectable gain (improvement {imp:+.4f} < {threshold}). ")
                           + "Inconclusive rather than refuted; set aside in favour of stronger leads.")
                    uncertain = ["Whether a small real effect exists: one split cannot resolve it."]
                if conf == "weak":
                    uncertain.append(f"The difference from the baseline is within noise ({z}).")
                pred = e["expected_metric"]
                if not success and pred is not None and n and abs(val - pred) <= 2 * math.sqrt(
                        max(val * (1 - val), 1e-6) / n):
                    uncertain.append(f"The researcher predicted {pred:.4f} and measured {val:.4f}: its own "
                                     "prediction held. It is judged here against the loop's fixed "
                                     "'beat the baseline' criterion, which may not be what this hypothesis asked.")
                if new_best:
                    if best is None:
                        uncertain.append("Only one candidate has beaten the baseline so far.")
                    best = e
                outcome_of_hyp.setdefault(e["hypothesis_id"], []).append(verdict)
            g.add(Node(did, "DECISION", f"{VERDICT_LABEL[verdict]}: {e['title'] or ''}"[:120], summary=why,
                       status=verdict, epistemic="invalid" if verdict == "invalid" else
                       ("benchmark" if verdict == "deprioritize" else "empirical"),
                       details={"before": before, "alternatives": alternatives, "new_evidence": [evid],
                                "affected": [hid] if hid else [], "why": why, "why_next": why_next,
                                "uncertain": uncertain,
                                "provenance": "derived from the evaluator's numeric outcome; next-step reasons "
                                              "are quoted from the researcher's recorded rationale"},
                       source_table="experiments", source_id=e["id"], created_at=e["finished_at"]))
            g.link(did, "depends_on", evid, "The evidence this decision was made on.", [evid], at=e["finished_at"])
            if hid and rel:
                g.link(did, rel, hid, why, [evid], at=e["finished_at"])
            if verdict == "adopt":
                cid = self.nid("conclusion", f"e{e['id']}")
                g.add(Node(cid, "CONCLUSION", f"Current best: {e['title']} (val {_fmt(e['val_score'])})",
                           summary=f"Held-out test {_fmt(e['test_score'])}. Provisional: holds until a better "
                                   "result arrives, and rests on the run's assumptions.",
                           status="current", confidence=self.g.nodes[evid].confidence, epistemic="empirical",
                           details={"experiment": e["id"], "val": e["val_score"], "test": e["test_score"]},
                           source_table="experiments", source_id=e["id"], created_at=e["finished_at"]))
                g.link(evid, "supports", cid, "Measured result behind this conclusion.", [evid], at=e["finished_at"])
                g.link(cid, "depends_on", did, "Adopted by this decision.", [evid], at=e["finished_at"])
                g.link(cid, "depends_on", a_proxy.id, "Chosen on validation scores alone.", at=e["finished_at"])
                g.link(cid, "depends_on", a_thresh.id, "Counts as a gain only via the threshold.", at=e["finished_at"])
                if conclusions:
                    prev = conclusions[-1]
                    g.nodes[prev].status = "superseded"
                    g.link(cid, "supersedes", prev, "A later experiment scored higher on validation.", [evid],
                           at=e["finished_at"])
                conclusions.append(cid)

        completed = [e for e in finished if e["status"] == "completed"]
        if completed and not conclusions:
            cid = self.nid("conclusion", "baseline")
            g.add(Node(cid, "CONCLUSION", "No tested hypothesis beat the baseline by the threshold",
                       summary="Every completed experiment missed the success threshold.", status="current",
                       epistemic="empirical", created_at=completed[-1]["finished_at"]))
            for e in completed:
                ev = self.nid("evidence", f"e{e['id']}")
                g.link(ev, "supports", cid, "This result missed the threshold.", [ev], at=e["finished_at"])
            g.link(cid, "depends_on", a_thresh.id, "Depends on what counts as a gain.")
            conclusions.append(cid)

        # Hypothesis status from its experiments' outcomes.
        for h in hyps:
            node = g.nodes[hyp_id[h["id"]]]
            outs = outcome_of_hyp.get(h["id"], [])
            running = any(e["status"] == "running" for e in by_hyp.get(h["id"], []))
            if "adopt" in outs:
                node.status, node.epistemic = "supported", "empirical"
            elif "deprioritize" in outs:
                node.status, node.epistemic = "supported_not_best", "benchmark"
            elif "reject" in outs:
                node.status, node.epistemic = "contradicted", "empirical"
            elif "set_aside" in outs:
                node.status, node.epistemic = "not_supported", "empirical"
            elif "invalid" in outs:
                node.status, node.epistemic = "untested_invalid", "invalid"
            elif running:
                node.status, node.epistemic = "in_progress", "not_investigated"
            else:
                node.status, node.epistemic = "not_investigated", "not_investigated"
            ev_ids = [self.nid("evidence", f"e{e['id']}") for e in by_hyp.get(h["id"], [])]
            confs = [g.nodes[i].confidence for i in ev_ids if i in g.nodes and g.nodes[i].confidence]
            node.confidence = confs[-1] if confs else None

        self._check_assumptions(a_proxy, a_thresh, base, completed, n, metric)
        self._open_questions(completed, conclusions, n, metric)
        # A cited parent that has no evidence yet (still running, interrupted) leaves no dangling edge.
        g.edges = {k: e for k, e in g.edges.items() if e.src in g.nodes and e.dst in g.nodes}
        return g

    # --- helpers --------------------------------------------------------------
    def _val_preds(self, e: dict | None) -> list[str] | None:
        """Per-example validation predictions of an experiment's last attempt, if they are on disk."""
        if not e or not self.work_dir:
            return None
        if e["id"] in self._pred_cache:
            return self._pred_cache[e["id"]]
        out = None
        for a in range(e["attempts"] or 1, 0, -1):
            f = self.work_dir / f"exp_{e['id']:04d}" / f"attempt_{a}" / "out" / "val_pred.csv"
            if f.exists():
                out = f.read_text(encoding="utf-8", errors="replace").split()
                break
        self._pred_cache[e["id"]] = out
        return out

    def cmp(self, new: dict, ref: dict | None, diff: float) -> tuple[str | None, str | None, dict]:
        """Strength of the difference between two experiments on the validation split.

        For accuracy with predictions on disk: exact paired McNemar test on the examples where the
        two disagree. Otherwise: an unpaired normal approximation (rougher, labelled as such).
        """
        if ref is None:
            return None, None, {}
        truth = self._val_truth()
        pa, pb = self._val_preds(new), self._val_preds(ref)
        if (self.domain and self.domain.metric == "accuracy" and truth and pa and pb
                and len(pa) == len(pb) == len(truth)):
            b = sum(1 for x, y, t in zip(pa, pb, truth) if x != t and y == t)   # only the reference right
            c = sum(1 for x, y, t in zip(pa, pb, truth) if x == t and y != t)   # only the new one right
            pval = _mcnemar_p(b, c)
            conf = "strong" if pval < 0.003 else "moderate" if pval < 0.05 else "weak"
            return conf, f"paired McNemar p={pval:.3g}", {"test": "paired McNemar (exact)", "p": round(pval, 5),
                                                            "only_new_right": c, "only_ref_right": b}
        conf, z = strength(diff, ref["val_score"], new["val_score"], self.n_val())
        if conf is None:
            return None, None, {}
        return conf, f"unpaired z={z}", {"test": "unpaired normal approximation", "z": z}

    def _val_truth(self) -> list[str] | None:
        if not hasattr(self, "_truth"):
            f = self.work_dir / "data" / "private" / "val_y.csv" if self.work_dir else None
            self._truth = f.read_text(encoding="utf-8").split() if f and f.exists() else None
        return self._truth

    def _round_key(self, e: dict, hyps: list[dict]):
        h = next((h for h in hyps if h["id"] == e["hypothesis_id"]), None)
        return (h["llm_call_id"] or -h["id"]) if h else None

    def _experiment(self, e: dict, hid: str | None, mid: str) -> str:
        cvs = self.m.rows("SELECT id, attempt, code_sha256 FROM code_versions WHERE experiment_id=? ORDER BY attempt",
                          e["id"])
        ws = (self.work_dir / f"exp_{e['id']:04d}") if self.work_dir else None
        title = "Baseline run" if e["is_baseline"] else f"#{e['id']} {e['title'] or ''}"
        node = self.g.add(Node(
            self.nid("exp", e["id"]), "EXPERIMENT", title[:120], status=e["status"],
            summary=(f"val {_fmt(e['val_score'])}, {e['attempts'] or 0} attempt(s)" if e["status"] == "completed"
                     else (e["error"] or e["status"])[:200]),
            details={"experiment_id": e["id"], "seed": e["seed"], "attempts": e["attempts"],
                     "code_versions": cvs, "system_revision": e["system_revision"],
                     "started_at": e["started_at"], "finished_at": e["finished_at"],
                     "workspace": str(ws) if ws else None, "error": e["error"]},
            source_table="experiments", source_id=e["id"], created_at=e["started_at"]))
        if hid:
            self.g.link(node.id, "tests", hid, "Runs the hypothesis's method in the sandbox.", at=e["started_at"])
        self.g.link(node.id, "depends_on", mid, "Implements this method.", at=e["started_at"])
        return node.id

    def _check_assumptions(self, a_proxy: Node, a_thresh: Node, base, completed, n, metric) -> None:
        if base and n:
            p = base["val_score"]
            se_diff = math.sqrt(2 * p * (1 - p) / n) if 0 <= p <= 1 else None
            if se_diff:
                thr = a_thresh.details["min_improvement"]
                a_thresh.details.update({"se_of_difference": round(se_diff, 4), "n_val": n})
                if thr < 2 * se_diff:
                    a_thresh.status = "questioned"
                    a_thresh.summary = (f"The threshold {thr} is below ~2 standard errors ({2 * se_diff:.4f}) of an unpaired "
                                        f"difference on {n} validation examples, so a 'success' near the threshold "
                                        "can be noise.")
                else:
                    a_thresh.status = "consistent"
                    a_thresh.summary = f"The threshold is above ~2 standard errors ({2 * se_diff:.4f})."
        pairs = [(e["val_score"], e["test_score"]) for e in completed
                 if e["val_score"] is not None and e["test_score"] is not None]
        if base and base["test_score"] is not None:
            pairs.append((base["val_score"], base["test_score"]))
        if len(pairs) >= 3:
            rho = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
            a_proxy.details.update({"spearman_val_test": rho, "n_experiments": len(pairs)})
            a_proxy.status = "consistent" if rho is not None and rho >= 0.5 else "questioned"
            a_proxy.summary = (f"Across {len(pairs)} scored runs, the rank correlation between validation and "
                               f"held-out test {metric} is {rho}. This checks the assumption; it does not prove it.")

    def _open_questions(self, completed, conclusions, n, metric) -> None:
        g = self.g
        succ = sorted((e for e in completed if e["success"]), key=lambda e: e["val_score"],
                      reverse=self.domain.higher_is_better if self.domain else True)
        if len(succ) >= 2:
            a, b = succ[0], succ[1]
            conf, z, zinfo = self.cmp(a, b, a["val_score"] - b["val_score"])
            if conf == "weak":
                oid = self.nid("open", f"top2_{a['id']}_{b['id']}")
                g.add(Node(oid, "OPEN_QUESTION", f"Which is better, #{a['id']} or #{b['id']}?",
                           summary=(f"Their validation {metric} differs by {abs(a['val_score'] - b['val_score']):.4f} "
                                    f"({z}), within the noise of one split. Next useful experiment: compare both "
                                    "with repeated cross-validation on the training split, or several seeds."),
                           status="open", details={"experiments": [a["id"], b["id"]], "comparison": zinfo,
                                                   "suggested_experiment": "repeated CV / multi-seed comparison"},
                           created_at=max(a["finished_at"], b["finished_at"])))
                for e in (a, b):
                    ev = self.nid("evidence", f"e{e['id']}")
                    g.link(ev, "motivates", oid, "Close score; the ranking is uncertain.", [ev], at=e["finished_at"])
        if self.g.nodes[self.nid("assumption", "threshold")].status == "questioned" and conclusions:
            oid = self.nid("open", "threshold")
            g.add(Node(oid, "OPEN_QUESTION", "Do the threshold-level 'successes' survive a noise check?",
                       summary="The success threshold is below the noise level of the validation split; successes "
                               "close to it should be re-checked before being relied on.",
                       status="open", created_at=g.nodes[conclusions[-1]].created_at))
            g.link(self.nid("assumption", "threshold"), "motivates", oid, "Assumption questioned by the noise check.")


def _mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _spearman(x: list[float], y: list[float]) -> float | None:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx, vy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return round(cov / math.sqrt(vx * vy), 3) if vx and vy else None


# --- persistence ------------------------------------------------------------
def ensure_schema(memory: Memory) -> None:
    memory.db.executescript(SCHEMA)
    cols = {r["name"] for r in memory.rows("PRAGMA table_info(graph_edges)")}
    if "stale" not in cols:      # graph tables created by an earlier version of this module
        memory.db.execute("ALTER TABLE graph_edges ADD COLUMN stale INTEGER DEFAULT 0")


def persist(memory: Memory, g: Graph) -> dict:
    """Upsert derived nodes/edges. Never deletes; derived rows no longer produced become `stale`
    (stale nodes stay visible with that status; stale edges are kept in the table but not loaded)."""
    ensure_schema(memory)
    now, rid = time.time(), g.run["id"]
    db = memory.db
    db.execute("BEGIN")
    try:
        for n in g.nodes.values():
            db.execute(
                "INSERT INTO graph_nodes (id, run_id, type, title, summary, status, confidence, epistemic, details_json,"
                " source_table, source_id, origin, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET type=excluded.type, title=excluded.title, summary=excluded.summary,"
                " status=excluded.status, confidence=excluded.confidence, epistemic=excluded.epistemic,"
                " details_json=excluded.details_json, updated_at=excluded.updated_at"
                " WHERE graph_nodes.origin='derived'",
                (n.id, rid, n.type, n.title, n.summary, n.status, n.confidence, n.epistemic,
                 json.dumps(n.details, default=str), n.source_table, n.source_id, n.origin,
                 n.created_at or now, now))
        for e in g.edges.values():
            db.execute(
                "INSERT INTO graph_edges (id, run_id, src, dst, rel, explanation, evidence_json, origin, created_at,"
                " updated_at) VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET"
                " explanation=excluded.explanation, evidence_json=excluded.evidence_json, updated_at=excluded.updated_at,"
                " stale=0 WHERE graph_edges.origin='derived'",
                (e.id, rid, e.src, e.dst, e.rel, e.explanation, json.dumps(e.evidence), e.origin,
                 e.created_at or now, now))
        ids = list(g.nodes)
        stale = db.execute(
            f"UPDATE graph_nodes SET status='stale', updated_at=? WHERE run_id=? AND origin='derived' "
            f"AND id NOT IN ({','.join('?' * len(ids))})", [now, rid, *ids]).rowcount if ids else 0
        eids = list(g.edges)
        if eids:
            db.execute(f"UPDATE graph_edges SET stale=1, updated_at=? WHERE run_id=? AND origin='derived' "
                       f"AND id NOT IN ({','.join('?' * len(eids))})", [now, rid, *eids])
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    return {"nodes": len(g.nodes), "edges": len(g.edges), "stale": stale}


def load(memory: Memory, run: dict) -> Graph:
    """The persisted graph for a run: derived plus any manually added nodes and edges."""
    ensure_schema(memory)
    g = Graph(run)
    for r in memory.rows("SELECT * FROM graph_nodes WHERE run_id=? ORDER BY created_at, id", run["id"]):
        g.nodes[r["id"]] = Node(r["id"], r["type"], r["title"], r["summary"] or "", r["status"] or "",
                                r["confidence"], r["epistemic"], json.loads(r["details_json"] or "{}"),
                                r["source_table"], r["source_id"], r["created_at"], r["origin"])
    for r in memory.rows("SELECT * FROM graph_edges WHERE run_id=? AND COALESCE(stale, 0)=0 "
                         "ORDER BY created_at, id", run["id"]):
        e = Edge(r["src"], r["rel"], r["dst"], r["explanation"] or "", json.loads(r["evidence_json"] or "[]"),
                 r["created_at"], r["origin"])
        if e.src in g.nodes and e.dst in g.nodes:
            g.edges[e.id] = e
    return g


def refresh(memory: Memory, run: dict, *, work_dir: Path | None = None, domain=None) -> Graph:
    """Rebuild the derived graph from the run's records, persist it, and return the full persisted graph."""
    persist(memory, Builder(memory, run, work_dir=work_dir, domain=domain).build())
    return load(memory, run)


def open_domain(name: str):
    from .domain import Domain
    try:
        return Domain(name)
    except (OSError, KeyError):
        return None


# --- queries used by the viewer and the narrative ------------------------------
def next_steps(g: Graph) -> list[dict]:
    """Useful next experiments, most valuable first. Negative results are not discounted."""
    out = []
    for n in g.nodes.values():
        if n.type == "OPEN_QUESTION" and n.status == "open":
            out.append({"node": n.id, "kind": "open_question", "title": n.title, "reason": n.summary, "rank": 0})
    for n in g.nodes.values():
        if n.type == "ASSUMPTION" and n.status in ("unverified", "questioned"):
            out.append({"node": n.id, "kind": "assumption", "title": f"Check assumption: {n.title}",
                        "reason": n.summary or "Not yet checked against any evidence.",
                        "rank": 1 if n.status == "questioned" else 3})
    for n in g.nodes.values():
        if n.type == "HYPOTHESIS" and n.status == "not_investigated":
            out.append({"node": n.id, "kind": "unexplored", "title": n.title,
                        "reason": f"Queued but never tested (priority {n.details.get('priority')}).",
                        "rank": 2 - float(n.details.get("priority") or 0) / 10})
        elif n.type == "HYPOTHESIS" and n.status == "untested_invalid":
            out.append({"node": n.id, "kind": "retry", "title": n.title,
                        "reason": "Its experiment failed, so the idea itself is still untested.", "rank": 2.5})
    return sorted(out, key=lambda x: x["rank"])


def trend(memory: Memory, run_id: int) -> list[dict]:
    rows = memory.rows("SELECT id, is_baseline, status, val_score, test_score, success, finished_at FROM experiments "
                       "WHERE run_id=? AND status='completed' ORDER BY finished_at, id", run_id)
    return rows


def compare(g: Graph, a: str, b: str) -> dict:
    """Side-by-side downstream lines of two nodes (e.g. two competing hypotheses)."""
    def line(nid):
        nodes, _ = g.trace(nid, "fwd")
        pick = lambda t: [g.nodes[i].__dict__ for i in sorted(nodes) if i in g.nodes and g.nodes[i].type == t]
        return {"root": g.nodes[nid].__dict__, "experiments": pick("EXPERIMENT"), "evidence": pick("EVIDENCE"),
                "decisions": pick("DECISION"), "hypotheses": [h for h in pick("HYPOTHESIS") if h["id"] != nid],
                "conclusions": pick("CONCLUSION")}
    na, _ = g.trace(a, "fwd")
    nb, _ = g.trace(b, "fwd")
    return {"a": line(a), "b": line(b), "shared": sorted((na & nb) - {a, b})}


# --- narrative --------------------------------------------------------------
def narrative(g: Graph) -> str:
    N = g.nodes
    by = lambda t: [n for n in N.values() if n.type == t]
    q = next(iter(by("QUESTION")), None)
    lines = [f"# Research narrative: {g.run.get('name')}", ""]
    if q:
        lines += [f"**Question.** {q.title}", "", q.summary, ""]
    lines += ["## Assumptions", ""]
    for a in by("ASSUMPTION"):
        lines.append(f"- **{a.title}** ({a.status}). {a.summary}".rstrip())
    current = [c for c in by("CONCLUSION") if c.status == "current"]
    lines += ["", "## Current conclusion", ""]
    for c in current:
        lines.append(f"**{c.title}.** {c.summary}")
        chain = g.path_to_question(c.id)
        if chain:
            lines += ["", "How it was reached (from the question to the conclusion):", ""]
            lines += [f"{i + 1}. `{N[x].type}` {N[x].title}" for i, x in enumerate(chain)]
        sup = [e for e in g.edges.values() if e.dst == c.id and e.rel == "supports"]
        if sup:
            lines += ["", "Supporting evidence: " + "; ".join(N[e.src].title for e in sup if e.src in N)]
    if not current:
        lines.append("No conclusion yet.")
    lines += ["", "## How the reasoning changed", ""]
    decisions = sorted(by("DECISION"), key=lambda n: n.created_at or 0)
    for d in decisions:
        det = d.details
        lines.append(f"### {d.title}")
        lines.append(f"- **Before:** {det.get('before', '')}")
        ev = [N[i].title for i in det.get("new_evidence", []) if i in N]
        if ev:
            lines.append(f"- **New evidence:** {'; '.join(ev)}")
        lines.append(f"- **Decision ({VERDICT_LABEL.get(d.status, d.status).lower()}, {d.epistemic}):** {d.summary}")
        if det.get("alternatives"):
            lines.append("- **Alternatives in the same round:** " + "; ".join(a["title"] for a in det["alternatives"]))
        if det.get("why_next"):
            lines.append("- **Next direction:** " + "; ".join(
                f"{w['title']} (\"{_short(w['rationale'], 160)}\")" for w in det["why_next"]))
        if det.get("uncertain"):
            lines.append("- **Still uncertain:** " + " ".join(det["uncertain"]))
        lines.append("")
    groups = [("Contradicted by experiment (empirical, not proven impossible)", "contradicted"),
              ("Not supported: no detectable gain (inconclusive, not refuted)", "not_supported"),
              ("Beat the baseline but deprioritized (benchmark ranking)", "supported_not_best"),
              ("Untested because the experiment was invalid", "untested_invalid"),
              ("Not yet investigated", "not_investigated")]
    lines += ["## Branches not taken (kept, not deleted)", ""]
    for label, st in groups:
        hs = [h for h in by("HYPOTHESIS") if h.status == st]
        if hs:
            lines.append(f"**{label}:**")
            lines += [f"- {h.title}" for h in hs]
            lines.append("")
    lines += ["## Open questions and next useful experiments", ""]
    for s in next_steps(g):
        lines.append(f"- **{s['title']}**: {s['reason']}")
    lines += ["", "---", "_Generated from the run's persisted records. Hypothesis rationales and next-step reasons are "
              "quoted from the researcher; everything else is derived from scores. Evidence strength uses an exact "
              "paired McNemar test where per-example predictions are on disk, otherwise an unpaired approximation, "
              "and always rests on a single validation split._", ""]
    return "\n".join(lines)
