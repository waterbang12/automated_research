"""CLI: python -m research {build-sandbox,new-domain,check-domain,run,report,show,permissions,graph} ..."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path

from . import config as config_mod
from .controller import ApprovalRequired, Controller
from .domain import IntegrityError
from .executor import ExecutorUnavailable
from .memory import Memory

ROOT = Path(__file__).resolve().parent.parent


def cmd_build(args) -> int:
    return subprocess.call(["docker", "build", "-t", args.image, str(ROOT / "sandbox")])


def cmd_run(args) -> int:
    cfg = config_mod.load(args.config)
    for k in ("max_experiments", "budget_usd", "workers"):
        if getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))
    if args.objective:
        cfg.objective = args.objective
    config_mod.check(cfg)
    ctl = Controller(cfg, approve=args.approve)
    try:
        asyncio.run(ctl.run())
    except (ApprovalRequired, ExecutorUnavailable, IntegrityError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted; completed experiments are saved, re-run the same command to resume", file=sys.stderr)
        return 130
    print_report(Memory(cfg.db_path), cfg.name)
    return 0


def print_report(mem: Memory, name: str, show_test: bool = True) -> None:
    run = mem.get_run(name)
    if not run:
        print(f"no run named {name!r}")
        return
    rid = run["id"]
    exps = mem.experiments(rid)
    print(f"\nRun {name!r}  domain={run['domain']}  revision={run['system_revision']}")
    print(f"Objective: {run['objective']}")
    print(f"{'id':>4} {'status':<11} {'val':>7} {'d_base':>8} {'test':>7} {'ok':>3} {'try':>3} {'parents':<9} title")
    for e in exps:
        val = f"{e['val_score']:.4f}" if e["val_score"] is not None else "-"
        imp = f"{e['improvement']:+.4f}" if e["improvement"] is not None else "-"
        test = f"{e['test_score']:.4f}" if e["test_score"] is not None and show_test else "-"
        title = "BASELINE" if e["is_baseline"] else (e["title"] or "")
        parents = ",".join(map(str, json.loads(e["parent_experiment_ids"] or "[]")))
        ok = "Y" if e["success"] else ""
        print(f"{e['id']:>4} {e['status']:<11} {val:>7} {imp:>8} {test:>7} {ok:>3} {e['attempts'] or 0:>3} "
              f"{parents:<9} {title[:70]}")
    done = [e for e in exps if e["status"] == "completed" and not e["is_baseline"]]
    if done:
        # "Best" follows the domain's direction: by `improvement` (positive is always better), not raw score.
        best = max(done, key=lambda e: e["improvement"])
        print(f"\nBest by validation: #{best['id']} {best['title']} val={best['val_score']:.4f} "
              f"held-out test={best['test_score']:.4f}")
    cost = mem.one("SELECT COUNT(*) n, COALESCE(SUM(cost_usd),0) c, COALESCE(SUM(input_tokens),0) i, "
                   "COALESCE(SUM(output_tokens),0) o FROM llm_calls WHERE run_id=?", rid)
    print(f"LLM calls: {cost['n']}  tokens in/out: {cost['i']}/{cost['o']}  cost: ${cost['c']:.4f}")
    perms = mem.rows("SELECT * FROM permission_requests WHERE run_id=? AND status='pending'", rid)
    if perms:
        print(f"Pending permission requests (not granted): {len(perms)} -> `python -m research permissions`")


def cmd_report(args) -> int:
    cfg = config_mod.load(args.config)
    print_report(Memory(cfg.db_path), cfg.name)
    return 0


def cmd_show(args) -> int:
    cfg = config_mod.load(args.config)
    mem = Memory(cfg.db_path)
    e = mem.one("SELECT e.*, h.title, h.rationale, h.method, h.parent_experiment_ids FROM experiments e "
                "LEFT JOIN hypotheses h ON h.id=e.hypothesis_id WHERE e.id=?", args.experiment)
    if not e:
        print("no such experiment")
        return 1
    for k in ("id", "status", "title", "rationale", "method", "parent_experiment_ids", "seed", "system_revision",
              "val_score", "test_score", "improvement", "success", "verdict", "analysis", "lessons", "error"):
        print(f"{k}: {e.get(k)}")
    for cv in mem.rows("SELECT * FROM code_versions WHERE experiment_id=? ORDER BY attempt", args.experiment):
        print(f"\n--- code attempt {cv['attempt']} sha256={cv['code_sha256'][:12]} llm_call={cv['llm_call_id']} ---")
        print(cv["code"])
    for x in mem.rows("SELECT * FROM executions WHERE experiment_id=? ORDER BY attempt", args.experiment):
        print(f"\n--- execution attempt {x['attempt']} exit={x['exit_code']} timeout={bool(x['timed_out'])} "
              f"{x['duration_s']}s image={x['image_id'][:19]} ---\n{x['stdout_tail']}\n{x['stderr_tail']}")
    return 0


def cmd_permissions(args) -> int:
    cfg = config_mod.load(args.config)
    mem = Memory(cfg.db_path)
    for p in mem.rows("SELECT * FROM permission_requests ORDER BY id"):
        print(f"#{p['id']} exp {p['experiment_id']} [{p['status']}] {p['request']}")
    print("Requests are never granted automatically. To grant one, change the [sandbox] config "
          "(and the sandbox image if packages are needed), then run with --approve.")
    return 0


def _run_graph(cfg):
    from . import graph as G
    mem = Memory(cfg.db_path)
    run = mem.get_run(cfg.name)
    if not run:
        raise SystemExit(f"no run named {cfg.name!r} in {cfg.db_path}")
    return mem, G.refresh(mem, run, work_dir=cfg.work_path if cfg.work_path.exists() else None,
                          domain=G.open_domain(run["domain"]))


def cmd_graph(args) -> int:
    from . import graph as G
    sys.stdout.reconfigure(errors="replace")      # LLM-written titles may not fit the console code page
    if args.action == "serve":
        from .viewer_server import serve
        srv = serve(Path(args.runs).resolve(), args.port)
        print(f"Research graph viewer: http://127.0.0.1:{args.port}/   (Ctrl+C to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    if not args.config:
        print("graph build/export need a config", file=sys.stderr)
        return 2
    mem, g = _run_graph(config_mod.load(args.config))
    if args.action == "build":
        from collections import Counter
        print(f"{len(g.nodes)} nodes, {len(g.edges)} edges: "
              + ", ".join(f"{k} {v}" for k, v in sorted(Counter(n.type for n in g.nodes.values()).items())))
        for s in G.next_steps(g)[:5]:
            print(f"  next: {s['title']} - {s['reason']}")
        return 0
    text = G.narrative(g) if args.format == "md" else json.dumps(g.to_dict(), indent=2, default=str)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


def cmd_new_domain(args) -> int:
    from .scaffold import new_domain
    for p in new_domain(args.name, Path(args.csv), args.target, args.task, args.objective):
        print(f"wrote {p}")
    print(f"\nNext: review domains/{args.name}/domain.toml (objective, min_improvement, contract), then\n"
          f"  python -m research check-domain {args.name}\n"
          f"  python -m research run configs/{args.name}.toml --max-experiments 2")
    return 0


def cmd_check_domain(args) -> int:
    from .config import SandboxConfig
    from .executor import DockerExecutor
    from .scaffold import check_domain
    sys.stdout.reconfigure(errors="replace")
    try:
        rows = asyncio.run(check_domain(args.name, DockerExecutor(SandboxConfig())))
    except ExecutorUnavailable as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    for label, ok, detail in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    failed = [r for r in rows if not r[1]]
    print("\n" + ("Domain is ready for a run." if not failed else f"{len(failed)} check(s) failed."))
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m research")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-sandbox", help="build the Docker image experiments run in")
    b.add_argument("--image", default="autoresearch-sandbox:1")
    r = sub.add_parser("run", help="start or resume a run")
    r.add_argument("config")
    r.add_argument("--max-experiments", dest="max_experiments", type=int)
    r.add_argument("--budget", dest="budget_usd", type=float)
    r.add_argument("--workers", type=int)
    r.add_argument("--objective")
    r.add_argument("--approve", action="store_true", help="confirm raised budget/limits/permissions")
    r.add_argument("-q", "--quiet", action="store_true")
    for name in ("report", "permissions"):
        p = sub.add_parser(name)
        p.add_argument("config")
    s = sub.add_parser("show", help="show one experiment: hypothesis, all code versions, outputs")
    s.add_argument("config")
    s.add_argument("experiment", type=int)
    nd = sub.add_parser("new-domain", help="scaffold a research problem from a CSV file")
    nd.add_argument("name")
    nd.add_argument("--csv", required=True, help="CSV with a header row")
    nd.add_argument("--target", required=True, help="column to predict")
    nd.add_argument("--task", choices=["classification", "regression"], help="default: inferred from the target")
    nd.add_argument("--objective", help="the research question; default: maximise accuracy / minimise RMSE")
    cd = sub.add_parser("check-domain", help="run a domain's prepare, baseline and scorer checks (no LLM)")
    cd.add_argument("name")
    gr = sub.add_parser("graph", help="research reasoning graph: build, export, or serve the viewer")
    gr.add_argument("action", choices=["build", "export", "serve"])
    gr.add_argument("config", nargs="?", help="run config (build/export)")
    gr.add_argument("--format", choices=["md", "json"], default="md", help="export format")
    gr.add_argument("-o", "--output", help="export to this file instead of stdout")
    gr.add_argument("--port", type=int, default=8765)
    gr.add_argument("--runs", default=str(ROOT / "runs"), help="directory with run databases (serve)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if getattr(args, "quiet", False) else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    return {"build-sandbox": cmd_build, "run": cmd_run, "report": cmd_report, "show": cmd_show,
            "permissions": cmd_permissions, "graph": cmd_graph, "new-domain": cmd_new_domain,
            "check-domain": cmd_check_domain}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
