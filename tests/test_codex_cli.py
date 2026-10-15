"""Codex CLI provider (`codex exec`), driven against a fake `codex` that speaks the same interface:
--output-schema FILE, --output-last-message FILE, --json events on stdout, prompt on stdin.

The real CLI's flags were checked against `codex exec --help` (codex-cli 0.162.1) and a live
unauthenticated call (which must surface as ProviderUnavailable); these tests cover the parsing,
budget and failure handling without credentials.
"""
import asyncio
import json
import subprocess
import sys

import pytest

from research import schemas
from research.config import LLMConfig, check
from research.llm import LLM, Budget, CodexCLIAdapter, LLMOutputError, ProviderUnavailable, resolve_codex
from research.memory import Memory

from fakes import make_config

ANALYSIS = {"verdict": "supported", "analysis": "fine", "lessons": ["x"]}

FAKE = r'''
import json, os, sys, time
args = sys.argv[1:]
val = lambda flag: args[args.index(flag) + 1]
prompt = sys.stdin.read()
mode = os.environ.get("FAKE_CODEX_MODE", "ok")
state = os.environ["FAKE_CODEX_STATE"]
calls = json.load(open(state)) if os.path.exists(state) else []
calls.append({"args": args, "prompt": prompt, "schema": json.load(open(val("--output-schema"))),
              "env_has_key": "OPENAI_API_KEY" in os.environ, "cwd": os.getcwd(), "pid": os.getpid()})
json.dump(calls, open(state, "w"))
ev = lambda **e: print(json.dumps(e), flush=True)
ev(type="thread.started", thread_id="t1")
ev(type="turn.started")
answer = %s
if mode == "hang":
    time.sleep(60)
if mode == "fail":
    ev(type="error", message="unexpected status 401 Unauthorized")
    ev(type="turn.failed", error={"message": "unexpected status 401 Unauthorized"})
    sys.exit(1)
if mode == "crash":
    print("panic: something broke", file=sys.stderr)
    sys.exit(3)
if mode == "malformed_then_ok" and len(calls) == 1:
    answer = "this is not json"
elif mode == "malformed_then_ok":
    answer = json.dumps(answer)
else:
    answer = json.dumps(answer)
print("a non-JSON log line the adapter must ignore", flush=True)
if mode == "events_only":      # no --output-last-message file; answer only in the event stream
    ev(type="item.completed", item={"id": "i1", "type": "agent_message", "text": answer})
else:
    open(val("--output-last-message"), "w").write(answer)
    ev(type="item.completed", item={"id": "i1", "type": "agent_message", "text": answer})
ev(type="turn.completed", usage={"input_tokens": 1200, "cached_input_tokens": 200, "output_tokens": 300})
''' % repr(ANALYSIS)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "fake_codex.py"
    script.write_text(FAKE)
    state = tmp_path / "calls.json"
    monkeypatch.setenv("FAKE_CODEX_STATE", str(state))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")

    def make(mode="ok", timeout=30, budget=5.0, **llm_over):
        monkeypatch.setenv("FAKE_CODEX_MODE", mode)
        cfg = make_config(tmp_path)
        cfg.llm = LLMConfig(provider="codex-cli", model="gpt-5", cli_timeout_s=timeout,
                            codex_path=str(script), **llm_over)
        cfg.pricing = {"gpt-5": (1.25, 10.0)}
        mem = Memory(tmp_path / "db.sqlite")
        llm = LLM(cfg, mem, 1, Budget(mem, 1, budget))
        adapter = llm._adapter("codex-cli")
        adapter.cmd = [sys.executable, str(script)]         # same interface, no Node needed in tests
        return llm, mem

    calls = lambda: json.loads(state.read_text()) if state.exists() else []
    return make, calls


def ask(llm):
    return asyncio.run(llm.structured(role="analyst", system="SYSTEM-TEXT", prompt="PROMPT-TEXT",
                                      schema=schemas.ANALYSIS, schema_name="analysis", context={}))


def test_structured_call_args_usage_and_cost(fake):
    make, calls = fake
    llm, mem = make()
    data, call_id = ask(llm)
    assert data == ANALYSIS
    c = calls()[0]
    a = c["args"]
    assert a[0] == "exec" and a[-1] == "-"                                   # prompt via stdin
    for flag in ("--skip-git-repo-check", "--ephemeral", "--json"):
        assert flag in a
    assert a[a.index("--sandbox") + 1] == "read-only" and a[a.index("--model") + 1] == "gpt-5"
    assert c["schema"] == schemas.ANALYSIS                                    # exact schema, from a file
    assert "SYSTEM-TEXT" in c["prompt"] and "PROMPT-TEXT" in c["prompt"]    # no system flag: prepended
    assert not c["env_has_key"]                                              # API key not inherited
    row = mem.one("SELECT * FROM llm_calls WHERE id=?", call_id)
    assert (row["provider"], row["input_tokens"], row["output_tokens"]) == ("codex-cli", 1200, 300)
    assert row["cost_usd"] == pytest.approx((1200 * 1.25 + 300 * 10.0) / 1e6)  # tokens x [pricing]


def test_answer_from_event_stream_when_no_output_file(fake):
    make, _ = fake
    llm, _ = make("events_only")
    assert ask(llm)[0] == ANALYSIS


def test_malformed_output_is_reasked_and_logged(fake):
    make, calls = fake
    llm, mem = make("malformed_then_ok")
    assert ask(llm)[0] == ANALYSIS
    rows = mem.rows("SELECT error FROM llm_calls ORDER BY id")
    assert len(rows) == 2 and rows[0]["error"].startswith("invalid output") and rows[1]["error"] is None
    assert "rejected by validation" in calls()[1]["prompt"]


def test_turn_failed_and_crash_stop_the_run(fake):
    make, _ = fake
    llm, mem = make("fail")
    with pytest.raises(ProviderUnavailable, match="401"):
        ask(llm)
    assert "ProviderUnavailable" in mem.one("SELECT error FROM llm_calls")["error"]
    llm, _ = make("crash")
    with pytest.raises(ProviderUnavailable, match="exit 3.*panic"):
        ask(llm)


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        import os
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_timeout_and_cancellation_kill_the_cli(fake):
    make, calls = fake
    llm, _ = make("hang", timeout=2)
    with pytest.raises(ProviderUnavailable, match="timed out"):
        ask(llm)
    assert not _alive(calls()[-1]["pid"])

    llm, _ = make("hang", timeout=60)

    async def cancel_soon():
        task = asyncio.create_task(llm.structured(role="analyst", system="s", prompt="p", schema=schemas.ANALYSIS,
                                                  schema_name="analysis", context={}))
        await asyncio.sleep(2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(cancel_soon())
    assert not _alive(calls()[-1]["pid"])


def test_budget_still_reserves_before_each_call(fake):
    make, calls = fake
    llm, _ = make(budget=0.0001)         # below one call's worst case (max_tokens x output price)
    from research.llm import BudgetExceeded
    with pytest.raises(BudgetExceeded):
        ask(llm)
    assert calls() == []                  # the CLI was never started


def test_options_and_config(fake, tmp_path):
    make, calls = fake
    llm, _ = make(codex_reasoning_effort="high", codex_inherit_api_key=True)
    ask(llm)
    c = calls()[-1]
    assert 'model_reasoning_effort="high"' in c["args"] and c["env_has_key"]
    cfg = make_config(tmp_path)
    cfg.llm = LLMConfig(provider="codex-cli", model="gpt-5")
    cfg.pricing = {"gpt-5": (1.25, 10.0)}
    check(cfg)                            # accepted by the existing provider/pricing validation


def test_resolve_codex_windows_shims(tmp_path):
    js = tmp_path / "npm" / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
    js.parent.mkdir(parents=True)
    js.write_text("")
    (tmp_path / "npm" / "codex.cmd").write_text("@echo off")
    cmd = resolve_codex(str(tmp_path / "npm" / "codex.cmd"))        # global npm layout
    assert cmd and cmd[-1] == str(js) and "node" in cmd[0].lower()
    local = tmp_path / "proj" / "node_modules"
    (local / ".bin").mkdir(parents=True)
    (local / ".bin" / "codex.cmd").write_text("@echo off")
    ljs = local / "@openai" / "codex" / "bin" / "codex.js"
    ljs.parent.mkdir(parents=True)
    ljs.write_text("")
    assert resolve_codex(str(local / ".bin" / "codex.cmd"))[-1] == str(ljs)   # project-local layout
    assert resolve_codex(str(js))[-1] == str(js)                              # codex.js directly
    assert resolve_codex(str(tmp_path / "missing" / "codex.cmd")) is None
