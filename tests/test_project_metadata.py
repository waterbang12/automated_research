"""Keeps project/ metadata honest: every evidence reference must exist, statuses must be used
consistently, and the files future sessions rely on must be present and linked."""
import ast
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "project"


def load(name):
    return yaml.safe_load((PROJECT / name).read_text(encoding="utf-8"))


def _test_names() -> dict[str, set[str]]:
    out = {}
    for f in (ROOT / "tests").glob("test_*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        out[f"tests/{f.name}"] = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    return out


TESTS = _test_names()


def check_evidence(ref: str) -> str | None:
    """Returns an error message, or None if the reference is valid."""
    kind, _, rest = ref.partition(":")
    if kind == "test":
        path, _, name = rest.partition("::")
        if path not in TESTS:
            return f"no test file {path}"
        if name != "*" and name not in TESTS[path]:
            return f"no test {name} in {path}"
    elif kind in ("file", "doc"):
        if not (ROOT / rest).exists():
            return f"no such path {rest}"
    elif kind == "commit":
        if not re.fullmatch(r"[0-9a-f]{7,40}", rest):
            return f"not a commit sha: {rest}"
    elif kind not in ("run", "manual"):
        return f"unknown evidence kind {kind!r}"
    return None


def all_evidence():
    caps, road = load("capabilities.yaml"), load("roadmap.yaml")
    for c in caps["implemented"]:
        for e in c["evidence"]:
            yield f"capabilities:{c['id']}", e
    for m in road["milestones"]:
        for e in m["evidence"]:
            yield f"roadmap:{m['id']}", e


def test_evidence_references_exist():
    errors = [f"{where}: {ref} -> {err}" for where, ref in all_evidence() if (err := check_evidence(ref))]
    assert not errors, "\n".join(errors)


def test_capability_statuses_are_backed():
    caps = load("capabilities.yaml")
    milestones = {m["id"] for m in load("roadmap.yaml")["milestones"]}
    ids = [c["id"] for c in caps["implemented"] + caps["proposed"]]
    assert len(ids) == len(set(ids)), "duplicate capability ids"
    for c in caps["implemented"]:
        assert c["status"] in ("verified", "tested"), c["id"]
        kinds = {e.partition(":")[0] for e in c["evidence"]}
        assert "test" in kinds, f"{c['id']}: implemented capabilities need at least one test"
        if c["status"] == "verified" and c["id"].split(".")[0] in ("provider", "sandbox", "viewer"):
            assert kinds & {"run", "manual", "doc"}, f"{c['id']}: verified external capability needs a real exercise"
    for p in caps["proposed"]:
        assert "status" not in p or p["status"] == "proposed", p["id"]
        assert p["milestone"] is None or p["milestone"] in milestones, p["id"]


def test_roadmap_is_well_formed():
    road = load("roadmap.yaml")
    ids = {m["id"] for m in road["milestones"]}
    for m in road["milestones"]:
        assert m["status"] in ("done", "in_progress", "next", "planned", "blocked"), m["id"]
        assert m["acceptance_criteria"], f"{m['id']} has no acceptance criteria"
        assert set(m["depends_on"]) <= ids, m["id"]
        if m["status"] == "done":
            assert m["evidence"], f"{m['id']} is done without evidence"
    for n in road["next_actions"]:
        assert n["acceptance_criteria"], n["id"]


def test_related_work_entries_are_verified():
    rw = load("related-work.yaml")
    for e in rw["entries"]:
        assert e["url"].startswith("https://") and e.get("verified_on"), e["id"]
        if e["kind"] == "paper":
            assert e["arxiv"] in e["url"] and e["authors"] and e["year"], e["id"]


def test_entry_points_exist_and_link():
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for f in ("vision.md", "state.md", "capabilities.yaml", "log.md", "roadmap.yaml", "decisions.md",
              "questions.md", "related-work.yaml"):
        assert (PROJECT / f).exists() and f"project/{f}" in agents, f
    assert (ROOT / "CLAUDE.md").read_text(encoding="utf-8").strip().startswith("@AGENTS.md")


@pytest.mark.parametrize("ref,ok", [("test:tests/test_graph.py::test_mcnemar", True),
                                    ("test:tests/test_graph.py::nope", False),
                                    ("file:research/nope.py", False), ("run:planted", True),
                                    ("magic:x", False)])
def test_evidence_checker(ref, ok):
    assert (check_evidence(ref) is None) == ok
