"""CLI: python -m research {build-sandbox,run,report,show,permissions} ..."""
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
        best = max(done, key=lambda e: e["val_score"])
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
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if getattr(args, "quiet", False) else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    return {"build-sandbox": cmd_build, "run": cmd_run, "report": cmd_report, "show": cmd_show,
            "permissions": cmd_permissions}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
