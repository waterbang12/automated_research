"""Exercise the real Anthropic/OpenAI SDK adapters against a local fake HTTP server.

This verifies request shape (structured-output parameters, model, effort) and
response/usage parsing without credentials or network access. It cannot verify
that the live APIs accept the request; that needs a real key (see README).
"""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from research import schemas
from research.llm import LLM, Budget
from research.memory import Memory

from fakes import make_config

ANALYSIS = {"verdict": "supported", "analysis": "fine", "lessons": ["x"]}


@pytest.fixture
def server():
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append((self.path, dict(self.headers), body))
            if "/chat/completions" in self.path:
                resp = {"id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-x-2026-01-01",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": json.dumps(ANALYSIS), "refusal": None}}],
                        "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}}
            else:
                resp = {"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
                        "content": [{"type": "thinking", "thinking": "", "signature": "s"},
                                    {"type": "text", "text": json.dumps(ANALYSIS)}],
                        "stop_reason": "end_turn", "stop_sequence": None,
                        "usage": {"input_tokens": 1000, "output_tokens": 200,
                                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", seen
    srv.shutdown()


def _call(tmp_path, provider, model):
    cfg = make_config(tmp_path)
    cfg.llm.provider, cfg.llm.model = provider, model
    cfg.pricing = {model: (4.0, 20.0)}
    mem = Memory(tmp_path / "db.sqlite")
    llm = LLM(cfg, mem, 1, Budget(mem, 1, 5.0))
    data, _ = asyncio.run(llm.structured(role="analyst", system="sys", prompt="hello", schema=schemas.ANALYSIS,
                                         schema_name="analysis", context={}))
    return data, mem.rows("SELECT * FROM llm_calls")[0]


def test_anthropic_adapter_request_and_usage(server, tmp_path, monkeypatch):
    url, seen = server
    monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    data, call = _call(tmp_path, "anthropic", "claude-opus-5-5")
    assert data == ANALYSIS
    path, headers, body = seen[0]
    assert body["model"] == "claude-opus-5-5" and body["system"] == "sys"
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": schemas.ANALYSIS}
    assert body["output_config"]["effort"] == "medium"
    assert body["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in headers.get("anthropic-beta", "")
    assert "thinking" not in body                     # Opus 5.5: thinking is always on; never send disabled
    assert call["cost_usd"] == pytest.approx((1000 * 4 + 200 * 20) / 1e6)


def test_openai_adapter_request_and_usage(server, tmp_path, monkeypatch):
    url, seen = server
    monkeypatch.setenv("OPENAI_BASE_URL", url + "/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    data, call = _call(tmp_path, "openai", "gpt-x")
    assert data == ANALYSIS
    _, _, body = seen[0]
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == schemas.ANALYSIS
    # Dated snapshot name in the response is billed at the configured model's price.
    assert call["cost_usd"] == pytest.approx((1000 * 4 + 200 * 20) / 1e6)


FAKE_CLAUDE = r'''
import json, sys
args = sys.argv[1:]
prompt = sys.stdin.read()
system = open(args[args.index("--system-prompt-file") + 1], encoding="utf-8").read()
json.dump({"args": args, "prompt": prompt, "system": system}, open(sys.argv[0] + ".seen", "w"))
print(json.dumps({"type": "result", "is_error": False, "stop_reason": "end_turn", "total_cost_usd": 0.0123,
                  "result": "", "structured_output": %s,
                  "modelUsage": {"claude-opus-5-5": {"inputTokens": 900, "cacheReadInputTokens": 100,
                                                     "cacheCreationInputTokens": 0, "outputTokens": 200,
                                                     "costUSD": 0.0123}}}))
''' % json.dumps(ANALYSIS).replace("null", "None")


def test_claude_cli_adapter_args_and_usage(tmp_path, monkeypatch):
    import sys
    from research import llm as llm_mod
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLAUDE)
    monkeypatch.setattr(llm_mod.shutil, "which", lambda name: sys.executable)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    cfg = make_config(tmp_path)
    cfg.llm.provider, cfg.llm.model = "claude-cli", "claude-opus-5-5"
    cfg.pricing = {"claude-opus-5-5": (4.0, 20.0)}
    mem = Memory(tmp_path / "db.sqlite")
    llm = LLM(cfg, mem, 1, Budget(mem, 1, 5.0))
    adapter = llm._adapter("claude-cli")
    adapter.cmd = [sys.executable, str(script)]
    assert "ANTHROPIC_API_KEY" not in adapter.env
    data, _ = asyncio.run(llm.structured(role="analyst", system="sys", prompt="hello", schema=schemas.ANALYSIS,
                                         schema_name="analysis", context={}))
    assert data == ANALYSIS
    seen = json.loads((tmp_path / "fake_claude.py.seen").read_text())
    args = seen["args"]
    assert seen["prompt"] == "hello" and seen["system"] == "sys"
    assert args[args.index("--model") + 1] == "claude-opus-5-5" and args[args.index("--tools") + 1] == ""
    assert json.loads(args[args.index("--json-schema") + 1]) == schemas.ANALYSIS
    call = mem.rows("SELECT * FROM llm_calls")[0]
    assert (call["input_tokens"], call["output_tokens"]) == (1000, 200)
    assert call["cost_usd"] == pytest.approx(0.0123)
