"""Research Reasoning Graph: derivation from real controller records, persistence, lineage queries,
epistemic labelling, backward compatibility, and the viewer API.

The records come from a full controller run with the fake executor (no generated code runs on
the host), so the graph is built from exactly the rows a real run persists.
"""
import asyncio
import json
import sqlite3
import threading
import urllib.request
from pathlib import Path

import pytest

from research import graph as G
from research.controller import Controller
from research.domain import Domain
from research.memory import Memory

from fakes import FakeExecutor, ScriptedAdapter, make_config, patch_baseline

# Baseline accuracy is 0.7 on 100 validation examples (see fakes.py).
PLAN = {
    "idea 1": ["acc=0.8"],            # beats baseline -> adopted
    "idea 2": ["acc=0.6"],            # clearly worse -> contradicted (paired test: strong)
    "idea 3": ["crash", "crash"],     # never scored -> invalid, hypothesis stays untested
    "idea 4": ["acc=0.9"],            # new best -> adopted, supersedes idea 1's conclusion
    "idea 5": ["acc=0.7"],            # ties the baseline -> not supported (inconclusive, not refuted)
}                                     # idea 6 is proposed but never run -> not investigated


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("graph")
    cfg = make_config(tmp, max_experiments=5, workers=1, max_fix_attempts=1)
    ctl = Controller(cfg, domain=patch_baseline(Domain("digits")), memory=Memory(cfg.db_path),
                     executor=FakeExecutor(), adapters={"anthropic": ScriptedAdapter(PLAN)})
    asyncio.run(ctl.run())
    return cfg


def build(cfg):
    mem = Memory(cfg.db_path)
    run = mem.get_run(cfg.name)
    return mem, G.refresh(mem, run, work_dir=cfg.work_path, domain=Domain("digits"))


def hyp(g, title):
    return next(n for n in g.nodes.values() if n.type == "HYPOTHESIS" and n.title == title)


def edges(g, rel=None, src=None, dst=None):
    return [e for e in g.edges.values()
            if (rel is None or e.rel == rel) and (src is None or e.src == src) and (dst is None or e.dst == dst)]


def test_statuses_distinguish_kinds_of_negative_result(recorded):
    _, g = build(recorded)
    assert hyp(g, "idea 4").status == "supported"
    assert hyp(g, "idea 1").status == "supported"          # adopted earlier, later superseded as best
    assert hyp(g, "idea 2").status == "contradicted" and hyp(g, "idea 2").epistemic == "empirical"
    assert hyp(g, "idea 3").status == "untested_invalid"
    assert hyp(g, "idea 5").status == "not_supported" and hyp(g, "idea 5").confidence == "weak"
    assert hyp(g, "idea 6").status == "not_investigated"
    # Never escalated to logical impossibility automatically.
    assert not edges(g, "rules_out")
    assert all(n.epistemic != "formal" for n in g.nodes.values())


def test_evidence_relationships(recorded):
    _, g = build(recorded)
    ev = lambda title: next(e.src for e in g.edges.values()
                            if e.rel in ("supports", "contradicts") and e.dst == hyp(g, title).id)
    assert edges(g, "supports", ev("idea 4"), hyp(g, "idea 4").id)
    assert edges(g, "contradicts", ev("idea 2"), hyp(g, "idea 2").id)
    # An invalid experiment is not evidence about its hypothesis.
    assert not [e for e in g.edges.values() if e.dst == hyp(g, "idea 3").id and e.rel in ("supports", "contradicts")]
    d3 = next(n for n in g.nodes.values() if n.type == "DECISION" and "idea 3" in n.title)
    assert d3.status == "invalid" and "untested" in d3.summary
    # Paired test on per-example predictions was used where available.
    e2 = g.nodes[ev("idea 2")]
    assert e2.details["vs_baseline"]["test"].startswith("paired McNemar")
    assert e2.confidence == "strong"
    # Edges carry timestamps, explanations, and evidence ids that resolve to nodes.
    for e in g.edges.values():
        assert e.created_at and e.explanation and e.rel in G.RELATIONS
        assert all(x in g.nodes for x in e.evidence)


def test_decisions_record_why_reasoning_changed(recorded):
    _, g = build(recorded)
    d1 = next(n for n in g.nodes.values() if n.type == "DECISION" and n.title.endswith("idea 1"))
    for k in ("before", "alternatives", "new_evidence", "affected", "why", "why_next", "uncertain"):
        assert k in d1.details
    assert d1.status == "adopt" and d1.details["affected"] == [hyp(g, "idea 1").id]
    assert [a["title"] for a in d1.details["alternatives"]] == ["idea 2"]       # same researcher round
    assert edges(g, "selects", d1.id, hyp(g, "idea 1").id)
    d2 = next(n for n in g.nodes.values() if n.type == "DECISION" and n.title.endswith("idea 2"))
    assert d2.status == "reject" and "not shown to be impossible" in d2.summary
    # The fake researcher cites the latest result; those citations become the "next direction".
    assert {w["title"] for w in d2.details["why_next"]} == {"idea 3", "idea 4"}
    assert edges(g, "inspired_by", hyp(g, "idea 4").id, d2.id)


def test_rejected_and_superseded_branches_are_preserved(recorded):
    _, g = build(recorded)
    concl = sorted((n for n in g.nodes.values() if n.type == "CONCLUSION"), key=lambda n: n.created_at)
    assert [c.status for c in concl] == ["superseded", "current"]
    assert edges(g, "supersedes", concl[1].id, concl[0].id)
    for t in ("idea 2", "idea 3", "idea 5", "idea 6"):
        assert hyp(g, t).id in g.nodes


def test_lineage_queries(recorded):
    _, g = build(recorded)
    current = next(n for n in g.nodes.values() if n.type == "CONCLUSION" and n.status == "current")
    path = g.path_to_question(current.id)
    types = [g.nodes[i].type for i in path]
    assert types[0] == "QUESTION" and path[-1] == current.id
    assert "EXPERIMENT" in types and "ASSUMPTION" not in types
    assert hyp(g, "idea 4").id in path
    back, back_edges = g.trace(current.id, "back")
    assert hyp(g, "idea 4").id in back and back_edges
    assert hyp(g, "idea 6").id not in back                 # never-run branch is not an ancestor
    fwd, _ = g.trace(hyp(g, "idea 2").id, "fwd")
    assert any(g.nodes[i].type == "EXPERIMENT" for i in fwd)
    assert {hyp(g, "idea 3").id, hyp(g, "idea 4").id} <= fwd, "ideas that built on this result"
    assert hyp(g, "idea 1").id not in g.trace(hyp(g, "idea 1").id, "fwd")[0] - {hyp(g, "idea 1").id}
    cmp = G.compare(g, hyp(g, "idea 4").id, hyp(g, "idea 2").id)
    assert cmp["a"]["evidence"] and cmp["b"]["evidence"]


def test_persistence_is_idempotent_and_ids_stable(recorded):
    mem, g1 = build(recorded)
    created = {n.id: n.created_at for n in g1.nodes.values()}
    _, g2 = build(recorded)
    assert set(g1.nodes) == set(g2.nodes) and set(g1.edges) == set(g2.edges)
    assert all(g2.nodes[i].created_at == created[i] for i in created)
    again = G.load(Memory(recorded.db_path), mem.get_run(recorded.name))
    assert set(again.nodes) == set(g2.nodes)


def test_manual_records_survive_and_dropped_derivations_go_stale(recorded):
    mem, g = build(recorded)
    rid = g.run["id"]
    q = next(n.id for n in g.nodes.values() if n.type == "QUESTION")
    mem.db.execute("INSERT INTO graph_nodes (id, run_id, type, title, status, origin, created_at, details_json) "
                   "VALUES ('manual:obs1', ?, 'OBSERVATION', 'Digit 8 is the most confused class', 'recorded', "
                   "'manual', 1, '{}')", (rid,))
    mem.db.execute("INSERT INTO graph_edges (id, run_id, src, dst, rel, explanation, origin, created_at) "
                   "VALUES ('manual:e1', ?, ?, 'manual:obs1', 'motivates', 'noted by hand', 'manual', 1)", (rid, q))
    mem.db.execute("INSERT INTO graph_nodes (id, run_id, type, title, status, origin, created_at, details_json) "
                   "VALUES ('r1:old:1', ?, 'HYPOTHESIS', 'from an older rule', 'supported', 'derived', 1, '{}')", (rid,))
    _, g2 = build(recorded)
    assert g2.nodes["manual:obs1"].status == "recorded" and "manual:obs1|motivates" not in g2.edges
    assert any(e.dst == "manual:obs1" for e in g2.edges.values())
    assert g2.nodes["r1:old:1"].status == "stale"          # kept, not deleted


def test_backward_compatible_with_existing_databases(recorded, tmp_path):
    # A database written before the graph existed: drop the graph tables and rebuild.
    db = tmp_path / "old.db"
    src, con = sqlite3.connect(recorded.db_path), sqlite3.connect(db)
    src.backup(con)          # includes rows still in the WAL file
    src.close()
    con.executescript("DROP TABLE IF EXISTS graph_nodes; DROP TABLE IF EXISTS graph_edges;"
                      # an earlier graph_edges layout without the `stale` column
                      "CREATE TABLE graph_edges (id TEXT PRIMARY KEY, run_id INTEGER, src TEXT, dst TEXT, rel TEXT,"
                      " explanation TEXT, evidence_json TEXT, origin TEXT, created_at REAL, updated_at REAL);")
    before = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("runs", "hypotheses", "experiments", "code_versions", "executions", "llm_calls")}
    con.close()
    mem = Memory(db)
    g = G.refresh(mem, mem.get_run(recorded.name), work_dir=recorded.work_path, domain=Domain("digits"))
    assert len(g.nodes) > 20
    after = {t: mem.one(f"SELECT COUNT(*) n FROM {t}")["n"] for t in before}
    assert after == before, "graph building never changes the loop's own records"
    # Works without the run workspace (e.g. a database copied elsewhere): unpaired fallback.
    g2 = G.Builder(mem, mem.get_run(recorded.name), work_dir=None, domain=Domain("digits")).build()
    e = next(n for n in g2.nodes.values() if n.type == "EVIDENCE" and n.status == "contradicts")
    assert e.confidence is None or e.details["vs_baseline"].get("test", "").startswith("unpaired")


def test_narrative_and_json_export(recorded):
    _, g = build(recorded)
    md = G.narrative(g)
    for heading in ("## Current conclusion", "## How the reasoning changed", "## Branches not taken",
                    "## Open questions and next useful experiments"):
        assert heading in md
    assert "Contradicted by experiment" in md and "idea 2" in md
    assert "Untested because the experiment was invalid" in md and "idea 3" in md
    assert "Not yet investigated" in md and "idea 6" in md
    data = json.loads(json.dumps(g.to_dict(), default=str))
    assert {"run", "nodes", "edges"} <= set(data)
    assert all({"flow_src", "flow_dst"} <= set(e) for e in data["edges"])
    kinds = {s["kind"] for s in G.next_steps(g)}
    assert {"unexplored", "retry"} <= kinds


def test_mcnemar():
    assert G._mcnemar_p(0, 0) == 1.0
    assert G._mcnemar_p(0, 10) == pytest.approx(2 / 1024)
    assert G._mcnemar_p(5, 5) == 1.0


def test_viewer_api(recorded):
    from research.viewer_server import serve
    srv = serve(recorded.db_path.parent, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    get = lambda p: urllib.request.urlopen(base + p, timeout=30)
    try:
        runs = json.load(get("/api/runs"))
        r = next(x for x in runs if x["name"] == recorded.name)
        q = f"db={r['db']}&run={r['id']}"
        g = json.load(get(f"/api/graph?{q}"))
        assert g["nodes"] and g["edges"] and g["next_steps"] and g["trend"]
        concl = next(n["id"] for n in g["nodes"] if n["type"] == "CONCLUSION" and n["status"] == "current")
        tr = json.load(get(f"/api/trace?{q}&id={concl}&dir=back"))
        assert concl in tr["nodes"] and tr["path"][-1] == concl
        exp = next(n["id"] for n in g["nodes"] if n["type"] == "EXPERIMENT" and n["source_id"] != 1)
        detail = json.load(get(f"/api/node?{q}&id={exp}"))
        assert detail["code_versions"] and detail["executions"]
        assert get(f"/api/export.md?{q}").read().decode().startswith("# Research narrative")
        assert b"Research Reasoning Graph" in get("/").read()
        with pytest.raises(urllib.error.HTTPError) as e:
            get(f"/api/graph?db=..%2F..%2Fsecret.db&run=1")
        assert e.value.code == 404
    finally:
        srv.shutdown()
