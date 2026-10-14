"""Local web server for the Research Reasoning Graph viewer (stdlib only, binds to 127.0.0.1).

Serves `viewer/` and a small JSON API over the run databases in `runs/*.db`. Each graph
request rebuilds the derived graph from the run's records (an idempotent upsert), so the
viewer always reflects the latest experiments, including those of a run still in progress.
"""
from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import graph as G
from .memory import Memory

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "viewer"


class Store:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir.resolve()

    def db_path(self, db: str) -> Path:
        p = (self.runs_dir / db).resolve()
        if p.parent != self.runs_dir or p.suffix != ".db" or not p.exists():
            raise LookupError(f"unknown database {db!r}")
        return p

    def runs(self) -> list[dict]:
        out = []
        for p in sorted(self.runs_dir.glob("*.db")):
            m = Memory(p)
            try:
                for r in m.rows("SELECT id, name, domain, objective, created_at FROM runs ORDER BY id"):
                    n = m.one("SELECT COUNT(*) n FROM experiments WHERE run_id=? AND is_baseline=0", r["id"])["n"]
                    out.append({**r, "db": p.name, "experiments": n})
            finally:
                m.close()
        return out

    def graph(self, db: str, run_id: int) -> tuple[Memory, G.Graph]:
        m = Memory(self.db_path(db))
        run = m.one("SELECT * FROM runs WHERE id=?", run_id)
        if not run:
            m.close()
            raise LookupError(f"no run {run_id} in {db}")
        work = self.runs_dir / run["name"]
        g = G.refresh(m, run, work_dir=work if work.exists() else None, domain=G.open_domain(run["domain"]))
        return m, g


def node_detail(m: Memory, g: G.Graph, nid: str) -> dict:
    n = g.nodes[nid]
    out = {"node": n.__dict__,
           "edges": [e for e in g.to_dict()["edges"] if nid in (e["src"], e["dst"])],
           "path_to_question": g.path_to_question(nid)}
    if n.type == "EXPERIMENT" and n.source_id:
        eid = n.source_id
        out["code_versions"] = m.rows("SELECT id, attempt, parent_version_id, code_sha256, explanation, code, "
                                      "llm_call_id, created_at FROM code_versions WHERE experiment_id=? ORDER BY attempt",
                                      eid)
        out["executions"] = m.rows("SELECT attempt, exit_code, timed_out, duration_s, image_id, command, stdout_tail, "
                                   "stderr_tail, created_at FROM executions WHERE experiment_id=? ORDER BY attempt", eid)
        out["llm_calls"] = m.rows("SELECT id, role, model, input_tokens, output_tokens, cost_usd, error, created_at "
                                  "FROM llm_calls WHERE experiment_id=? ORDER BY id", eid)
    return out


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def js(self, obj, code=200, extra=None):
            self.send(code, json.dumps(obj, default=str).encode(), "application/json", extra)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path.startswith("/api/"):
                    return self.api(u.path[5:], q)
                return self.static(u.path)
            except LookupError as e:
                return self.js({"error": str(e)}, 404)
            except Exception as e:      # surface errors to the viewer instead of a dropped connection
                return self.js({"error": f"{type(e).__name__}: {e}"}, 500)

        def static(self, path: str):
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            p = (VIEWER / rel).resolve()
            if VIEWER.resolve() not in p.parents or not p.is_file():
                return self.send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            if p.suffix == ".js":
                ctype = "text/javascript"
            return self.send(200, p.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text") else ""))

        def api(self, name: str, q: dict):
            if name == "runs":
                return self.js(store.runs())
            m, g = store.graph(q["db"], int(q["run"]))
            try:
                if name == "graph":
                    return self.js({**g.to_dict(), "next_steps": G.next_steps(g), "trend": G.trend(m, g.run["id"])})
                if name == "node":
                    if q["id"] not in g.nodes:
                        raise LookupError(f"no node {q['id']}")
                    return self.js(node_detail(m, g, q["id"]))
                if name == "trace":
                    nodes, edges = g.trace(q["id"], q.get("dir", "back"))
                    return self.js({"nodes": sorted(nodes), "edges": sorted(edges),
                                    "path": g.path_to_question(q["id"]) if q.get("dir", "back") == "back" else []})
                if name == "compare":
                    return self.js(G.compare(g, q["a"], q["b"]))
                if name == "export.json":
                    return self.js(g.to_dict(), extra={"content-disposition":
                                                       f'attachment; filename="{g.run["name"]}-graph.json"'})
                if name == "export.md":
                    return self.send(200, G.narrative(g).encode(), "text/markdown; charset=utf-8",
                                     {"content-disposition": f'attachment; filename="{g.run["name"]}-narrative.md"'})
                raise LookupError(f"unknown endpoint {name}")
            finally:
                m.close()

    return Handler


def serve(runs_dir: Path, port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(Store(runs_dir)))
