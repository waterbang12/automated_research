"""LLM access: provider adapters, structured output + validation, cost accounting, budget.

Every call reserves its worst-case cost against the budget *before* it is sent,
so concurrent workers can never jointly overspend. Actual cost is recorded in
`llm_calls` (which is also what resume reads to know what was already spent).
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Protocol

from .config import Config
from .memory import Memory
from .schemas import ValidationError, validate


class BudgetExceeded(Exception):
    pass


class LLMOutputError(Exception):
    pass


class ProviderUnavailable(RuntimeError):
    """The provider cannot serve calls right now (usage limit, auth, outdated CLI). Stops the run."""


class Usage:
    def __init__(self, input_tokens: int = 0, output_tokens: int = 0, model: str = "",
                 cost_usd: float | None = None):
        self.input_tokens, self.output_tokens, self.model = input_tokens, output_tokens, model
        self.cost_usd = cost_usd        # set when the provider reports its own cost


class Adapter(Protocol):
    async def complete(self, *, role: str, model: str, system: str, prompt: str, schema: dict,
                       schema_name: str, context: dict) -> tuple[str, Usage]: ...


class AnthropicAdapter:
    """Claude via the official SDK, using output_config.format for JSON-schema outputs."""

    def __init__(self, cfg: Config):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.AsyncAnthropic(max_retries=3)   # SDK retries 408/409/429/5xx
        self.cfg = cfg.llm
        self.fallbacks = cfg.llm.anthropic_fallbacks

    async def complete(self, *, role, model, system, prompt, schema, schema_name, context):
        kwargs = dict(
            model=model, max_tokens=self.cfg.max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.cfg.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        if self.fallbacks:
            try:
                resp = await self.client.beta.messages.create(
                    **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            except self.anthropic.BadRequestError as e:
                if "fallback" not in str(e).lower():
                    raise
                self.fallbacks = False      # account/model does not support it; continue without
                resp = await self.client.messages.create(**kwargs)
        else:
            resp = await self.client.messages.create(**kwargs)
        u = resp.usage
        inp = u.input_tokens + int(1.25 * (u.cache_creation_input_tokens or 0)) + (u.cache_read_input_tokens or 0)
        usage = Usage(inp, u.output_tokens, getattr(resp, "model", model) or model)
        if resp.stop_reason == "refusal":
            raise LLMOutputError(f"model refused ({getattr(resp, 'stop_details', None)})")
        text = "".join(b.text for b in resp.content if b.type == "text")
        if resp.stop_reason == "max_tokens":
            text += "\n<<TRUNCATED: hit max_tokens>>"
        return text, usage


class OpenAIAdapter:
    """OpenAI Chat Completions with strict json_schema response_format."""

    def __init__(self, cfg: Config):
        import openai
        self.client = openai.AsyncOpenAI(max_retries=3)
        self.cfg = cfg.llm

    async def complete(self, *, role, model, system, prompt, schema, schema_name, context):
        resp = await self.client.chat.completions.create(
            model=model, max_completion_tokens=self.cfg.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": schema_name, "schema": schema, "strict": True}},
        )
        usage = Usage(resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.model or model)
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None):
            raise LLMOutputError(f"model refused: {msg.refusal}")
        return msg.content or "", usage


class ClaudeCLIAdapter:
    """Headless Claude Code (`claude -p`), billed to the logged-in Claude subscription.

    Tools are disabled, so the model only returns text. The prompt goes over stdin and the
    system prompt through a temp file (Windows caps command lines at ~32k chars). The CLI's
    `total_cost_usd` is the API-equivalent cost, which is what the run budget counts.
    """

    def __init__(self, cfg: Config):
        exe = shutil.which("claude")
        if not exe:
            raise RuntimeError("provider 'claude-cli' needs Claude Code on PATH (`claude` not found)")
        self.cmd, self.cfg = [exe], cfg.llm
        # Without an API key in the environment the CLI uses the subscription login.
        self.env = {k: v for k, v in os.environ.items()
                    if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE")}
        self.cwd = tempfile.mkdtemp(prefix="research-claude-cli-")   # keep project CLAUDE.md/settings out

    async def complete(self, *, role, model, system, prompt, schema, schema_name, context):
        sys_file = Path(self.cwd) / f"system-{os.getpid()}-{id(prompt)}.txt"
        sys_file.write_text(system, encoding="utf-8")
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.cmd, "-p", "--model", model, "--effort", self.cfg.effort, "--tools", "",
                "--output-format", "json", "--json-schema", json.dumps(schema),
                "--system-prompt-file", str(sys_file), "--no-session-persistence", "--strict-mcp-config",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd, env=self.env)
            try:
                out, err = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")),
                                                  timeout=self.cfg.cli_timeout_s)
            except BaseException:          # timeout or cancellation: never leave the CLI running
                proc.kill()
                await proc.wait()
                raise
        except asyncio.TimeoutError:
            raise ProviderUnavailable(f"claude -p timed out after {self.cfg.cli_timeout_s}s")
        finally:
            sys_file.unlink(missing_ok=True)
        try:
            res = json.loads(out.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            raise ProviderUnavailable(f"claude -p exit {proc.returncode}: "
                               f"{(err or out).decode('utf-8', 'replace')[-2000:]}")
        mu = res.get("modelUsage") or {}
        inp = sum(m.get("inputTokens", 0) + m.get("cacheReadInputTokens", 0)
                  + m.get("cacheCreationInputTokens", 0) for m in mu.values())
        usage = Usage(inp, sum(m.get("outputTokens", 0) for m in mu.values()),
                      model if model in mu or not mu else max(mu, key=lambda k: mu[k].get("costUSD", 0)),
                      float(res.get("total_cost_usd") or 0.0))
        if res.get("is_error"):
            # Usage limits, auth and API errors: stop rather than burn retries.
            raise ProviderUnavailable(f"claude -p error ({res.get('subtype')}): {res.get('result')}")
        if res.get("stop_reason") == "refusal":
            raise LLMOutputError("model refused")
        if res.get("structured_output") is not None:
            return json.dumps(res["structured_output"]), usage
        return res.get("result") or "", usage      # let schema validation report the problem


def resolve_codex(configured: str = "") -> list[str] | None:
    """Command prefix that launches the Codex CLI, or None if it cannot be found.

    npm installs `codex` on Windows as a .cmd shim, which Python cannot start without cmd.exe
    (and cmd.exe re-parses arguments). The shim only runs `node .../@openai/codex/bin/codex.js`,
    so run that directly. `configured` may be an exe, a .cmd shim or the codex.js path.
    """
    path = configured or shutil.which("codex")
    if not path:
        return None
    p = Path(path)
    if p.suffix.lower() == ".js":
        node = shutil.which("node")
        return [node, str(p)] if node else None
    if p.suffix.lower() in (".cmd", ".bat", ".ps1"):
        # global install: %APPDATA%\npm\codex.cmd; local install: node_modules\.bin\codex.cmd
        rel = Path("@openai") / "codex" / "bin" / "codex.js"
        js = next((c for c in (p.parent / "node_modules" / rel, p.parent.parent / rel) if c.exists()), None)
        node = (p.parent / "node.exe") if (p.parent / "node.exe").exists() else shutil.which("node")
        return [str(node), str(js)] if js and node else None
    return [str(p)]


class CodexCLIAdapter:
    """OpenAI Codex CLI run non-interactively (`codex exec`).

    Each call runs in a fresh, empty temp directory with a read-only sandbox, no persisted
    session (--ephemeral) and no git requirement. The JSON schema goes in a file
    (--output-schema), the final message comes back through --output-last-message, and usage
    is read from the --json event stream (`turn.completed`). The CLI reports tokens, not money:
    the budget charges tokens x [pricing], an API-equivalent estimate that is not what a
    ChatGPT plan bills. Codex has no system-prompt flag, so the system text is prepended.
    """

    PREAMBLE = ("You are being called as a component of an automated research loop. Do not run commands, "
                "read or write files, or ask questions. Reply only with the final answer as JSON that matches "
                "the provided output schema.")

    def __init__(self, cfg: Config):
        cmd = resolve_codex(cfg.llm.codex_path)
        if not cmd:
            raise RuntimeError("provider 'codex-cli' needs the Codex CLI (npm i -g @openai/codex), or set "
                               "[llm] codex_path")
        self.cmd, self.cfg = cmd, cfg.llm
        drop = {"CODEX_MANAGED_BY_NPM"} | (set() if cfg.llm.codex_inherit_api_key else {"OPENAI_API_KEY", "CODEX_API_KEY"})
        self.env = {k: v for k, v in os.environ.items() if k not in drop}

    def args(self, model: str, schema_file: Path, out_file: Path, workdir: Path) -> list[str]:
        a = [*self.cmd, "exec", "--skip-git-repo-check", "--ephemeral", "--sandbox", "read-only",
             "--json", "--color", "never", "--output-schema", str(schema_file),
             "--output-last-message", str(out_file), "--cd", str(workdir)]
        if model and model != "default":
            a += ["--model", model]
        if self.cfg.codex_reasoning_effort:
            a += ["-c", f'model_reasoning_effort="{self.cfg.codex_reasoning_effort}"']
        return a + ["-"]                    # prompt from stdin: no command-line length limit

    async def complete(self, *, role, model, system, prompt, schema, schema_name, context):
        work = Path(tempfile.mkdtemp(prefix="research-codex-"))
        schema_file, out_file = work / f"{schema_name}.schema.json", work / "last_message.txt"
        schema_file.write_text(json.dumps(schema), encoding="utf-8")
        text = f"{self.PREAMBLE}\n\n# Instructions\n{system}\n\n# Task\n{prompt}"
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.args(model, schema_file, out_file, work), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=str(work), env=self.env)
            try:
                out, err = await asyncio.wait_for(proc.communicate(text.encode("utf-8")),
                                                  timeout=self.cfg.cli_timeout_s)
            except BaseException:          # timeout or cancellation: never leave the CLI running
                proc.kill()
                await proc.wait()
                raise
            last = out_file.read_text(encoding="utf-8", errors="replace") if out_file.exists() else None
        except asyncio.TimeoutError:
            raise ProviderUnavailable(f"codex exec timed out after {self.cfg.cli_timeout_s}s")
        except OSError as e:
            raise ProviderUnavailable(f"could not start codex: {e}")
        finally:
            shutil.rmtree(work, ignore_errors=True)
        events = []
        for line in out.decode("utf-8", "replace").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue                    # the CLI may print non-JSON lines; only events matter
        u = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
        for ev in events:
            if ev.get("type") == "turn.completed":
                for k in u:
                    u[k] += int((ev.get("usage") or {}).get(k) or 0)
        usage = Usage(u["input_tokens"], u["output_tokens"], model)
        failed = next((ev for ev in reversed(events) if ev.get("type") == "turn.failed"), None)
        if failed is not None:
            # Auth, usage limits, network: stop the run rather than burn retries (resume later).
            raise ProviderUnavailable(f"codex exec failed: {(failed.get('error') or {}).get('message', failed)}"[:1000])
        if proc.returncode != 0:
            errs = [ev.get("message") for ev in events if ev.get("type") == "error"]
            detail = errs[-1] if errs else err.decode("utf-8", "replace")[-1000:]
            raise ProviderUnavailable(f"codex exec exit {proc.returncode}: {detail}")
        if last is None:                    # fall back to the last agent message in the event stream
            msgs = [ev["item"].get("text", "") for ev in events if ev.get("type") == "item.completed"
                    and (ev.get("item") or {}).get("type") == "agent_message"]
            last = msgs[-1] if msgs else ""
        return last.strip(), usage          # schema validation (and re-asking) happens in LLM.structured


class OfflineAdapter:
    """Deterministic, zero-cost stand-in for an LLM, supplied by the domain.

    It receives the same structured history the prompts are built from, so it
    still adapts to recorded results; it is a fixture for running the full
    pipeline without credentials, not a source of research ideas.
    """

    def __init__(self, fixture):
        self.fixture = fixture

    async def complete(self, *, role, model, system, prompt, schema, schema_name, context):
        fn = getattr(self.fixture, role)
        return json.dumps(fn(context)), Usage(0, 0, "offline")


def credentials_missing(cfg: Config) -> list[str]:
    need = {cfg.llm.for_role(r).provider for r in ("researcher", "coder", "analyst")}
    missing = []
    if "anthropic" in need and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        missing.append("ANTHROPIC_API_KEY")
    if "openai" in need and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if "claude-cli" in need and not shutil.which("claude"):
        missing.append("the `claude` CLI on PATH (logged in)")
    if "codex-cli" in need and not resolve_codex(cfg.llm.codex_path):
        missing.append("the Codex CLI (npm i -g @openai/codex, then `codex login`), or [llm] codex_path")
    return missing


class Budget:
    def __init__(self, memory: Memory, run_id: int, limit_usd: float):
        self.memory, self.run_id, self.limit = memory, run_id, limit_usd
        self.reserved = 0.0
        self.lock = asyncio.Lock()

    def spent(self) -> float:
        return self.memory.total_cost(self.run_id)

    def remaining(self) -> float:
        return self.limit - self.spent() - self.reserved

    async def reserve(self, amount: float) -> None:
        async with self.lock:
            if self.spent() + self.reserved + amount > self.limit + 1e-12:
                raise BudgetExceeded(
                    f"next call could cost up to ${amount:.4f}; spent ${self.spent():.4f} "
                    f"+ reserved ${self.reserved:.4f} of ${self.limit:.2f}")
            self.reserved += amount

    def release(self, amount: float) -> None:
        self.reserved = max(0.0, self.reserved - amount)


class LLM:
    def __init__(self, cfg: Config, memory: Memory, run_id: int, budget: Budget,
                 adapters: dict[str, Adapter] | None = None, fixture=None):
        self.cfg, self.memory, self.run_id, self.budget = cfg, memory, run_id, budget
        self.adapters: dict[str, Adapter] = dict(adapters or {})
        self.fixture = fixture

    def _adapter(self, provider: str) -> Adapter:
        if provider not in self.adapters:
            if provider == "anthropic":
                self.adapters[provider] = AnthropicAdapter(self.cfg)
            elif provider == "openai":
                self.adapters[provider] = OpenAIAdapter(self.cfg)
            elif provider == "claude-cli":
                self.adapters[provider] = ClaudeCLIAdapter(self.cfg)
            elif provider == "codex-cli":
                self.adapters[provider] = CodexCLIAdapter(self.cfg)
            elif provider == "offline":
                if self.fixture is None:
                    raise RuntimeError("provider 'offline' needs a domain with offline_fixture.py")
                self.adapters[provider] = OfflineAdapter(self.fixture)
            else:
                raise ValueError(provider)
        return self.adapters[provider]

    def price(self, model: str, inp: int, out: int) -> float:
        if model == "offline":
            return 0.0
        # Unknown model (e.g. a server-side fallback): charge the most expensive configured rate.
        p_in, p_out = self.cfg.pricing.get(model) or max(self.cfg.pricing.values(), key=lambda p: p[1])
        return (inp * p_in + out * p_out) / 1e6

    def worst_case(self, provider: str, model: str, text: str) -> float:
        if provider == "offline":
            return 0.0
        est_input = len(text) // 3 + 500          # generous chars-per-token estimate
        return self.price(model, est_input, self.cfg.llm.max_tokens)

    async def structured(self, *, role: str, system: str, prompt: str, schema: dict, schema_name: str,
                         context: dict | None = None, experiment_id: int | None = None,
                         check: Callable[[Any], None] | None = None) -> tuple[Any, int]:
        """Return (validated object, llm_call_id). Re-asks on invalid output up to the retry limit."""
        rm = self.cfg.llm.for_role(role)
        adapter = self._adapter(rm.provider)
        ask, last_err = prompt, None
        for _ in range(self.cfg.llm_validation_retries + 1):
            reservation = self.worst_case(rm.provider, rm.model, system + ask)
            await self.budget.reserve(reservation)
            try:
                text, usage = await adapter.complete(role=role, model=rm.model, system=system, prompt=ask,
                                                     schema=schema, schema_name=schema_name,
                                                     context=context or {})
            except Exception as e:
                self.memory.add_llm_call(run_id=self.run_id, experiment_id=experiment_id, role=role,
                                         provider=rm.provider, model=rm.model, system=system, prompt=ask,
                                         response=None, input_tokens=0, output_tokens=0, cost_usd=0.0,
                                         error=f"{type(e).__name__}: {e}")
                raise
            finally:
                self.budget.release(reservation)
            # OpenAI reports dated snapshot names; bill those at the configured model's rate.
            # An unknown Anthropic model can only be a server-side fallback: bill at the max rate.
            billed = usage.model if usage.model in self.cfg.pricing or rm.provider != "openai" else rm.model
            if rm.provider == "offline":
                cost = 0.0
            elif usage.cost_usd is not None:
                cost = usage.cost_usd
            else:
                cost = self.price(billed, usage.input_tokens, usage.output_tokens)
            call_id = self.memory.add_llm_call(
                run_id=self.run_id, experiment_id=experiment_id, role=role, provider=rm.provider,
                model=usage.model or rm.model, system=system, prompt=ask, response=text,
                input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, cost_usd=cost, error=None)
            try:
                data = json.loads(text)
                validate(data, schema)
                if check:
                    check(data)
                return data, call_id
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                last_err = e
                self.memory.db.execute("UPDATE llm_calls SET error=? WHERE id=?", (f"invalid output: {e}", call_id))
                ask = (f"{prompt}\n\n---\nYour previous answer was rejected by validation: {e}\n"
                       f"Previous answer (truncated):\n{text[:4000]}\n\nReturn a corrected answer that satisfies the schema.")
        raise LLMOutputError(f"{role}: no valid output after retries: {last_err}")
