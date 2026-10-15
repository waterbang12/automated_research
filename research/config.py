"""TOML configuration. Secrets never live here; adapters read them from env vars."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class RoleModel:
    provider: str
    model: str


@dataclass
class LLMConfig:
    provider: str = "anthropic"          # anthropic | openai | claude-cli | codex-cli | offline
    model: str = "claude-opus-5-5"
    effort: str = "medium"               # Anthropic output_config.effort / claude --effort
    max_tokens: int = 16000              # not settable via claude-cli; still sizes the budget reservation
    cli_timeout_s: int = 600             # per `claude -p` / `codex exec` call
    codex_path: str = ""                 # codex-cli: exe, npm .cmd shim or codex.js; default: `codex` on PATH
    codex_reasoning_effort: str = ""     # codex-cli: passed as -c model_reasoning_effort="..." when set
    codex_inherit_api_key: bool = False  # codex-cli: pass OPENAI_API_KEY/CODEX_API_KEY through (API billing)
    anthropic_fallbacks: bool = True     # server-side refusal fallback (beta)
    roles: dict[str, RoleModel] = field(default_factory=dict)  # researcher/coder/analyst overrides

    def for_role(self, role: str) -> RoleModel:
        return self.roles.get(role, RoleModel(self.provider, self.model))


@dataclass
class SandboxConfig:
    image: str = "autoresearch-sandbox:1"
    timeout_s: int = 120
    memory: str = "1g"
    cpus: float = 1.0
    pids: int = 128
    network: str = "none"
    max_output_bytes: int = 20_000       # stdout/stderr tail kept per execution


@dataclass
class Config:
    name: str
    domain: str
    objective: str = ""                  # empty -> domain default
    db: str = ""
    workdir: str = ""
    max_experiments: int = 6
    budget_usd: float = 1.0
    workers: int = 2
    hypotheses_per_round: int = 2
    max_fix_attempts: int = 2            # code-repair attempts after a failed execution
    llm_validation_retries: int = 2      # re-asks after schema-invalid output
    llm_analysis: bool = True
    llm: LLMConfig = field(default_factory=LLMConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    pricing: dict[str, tuple[float, float]] = field(default_factory=dict)  # USD per 1M tokens (in, out)
    root: Path = Path(".")

    @property
    def db_path(self) -> Path:
        return self.root / (self.db or f"runs/{self.name}.db")

    @property
    def work_path(self) -> Path:
        return self.root / (self.workdir or f"runs/{self.name}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("root")
        return d


def load(path: str | Path, root: Path | None = None) -> Config:
    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    run = raw.get("run", {})
    llm_raw = dict(raw.get("llm", {}))
    roles = {k: RoleModel(**v) for k, v in llm_raw.pop("roles", {}).items()}
    cfg = Config(
        **run,
        llm=LLMConfig(**llm_raw, roles=roles),
        sandbox=SandboxConfig(**raw.get("sandbox", {})),
        pricing={k: (float(v[0]), float(v[1])) for k, v in raw.get("pricing", {}).items()},
        root=root or Path(__file__).resolve().parent.parent,
    )
    check(cfg)
    return cfg


def check(cfg: Config) -> None:
    if cfg.workers < 1 or cfg.workers > 8:
        raise ValueError("workers must be 1..8")
    if cfg.budget_usd < 0 or cfg.max_experiments < 0:
        raise ValueError("budget_usd and max_experiments must be >= 0")
    for role in ("researcher", "coder", "analyst"):
        rm = cfg.llm.for_role(role)
        if rm.provider not in ("anthropic", "openai", "claude-cli", "codex-cli", "offline"):
            raise ValueError(f"unknown provider {rm.provider!r}")
        # Refuse to spend money on a model we cannot price: the budget would be meaningless.
        if rm.provider != "offline" and rm.model not in cfg.pricing:
            raise ValueError(f"no [pricing] entry for model {rm.model!r}; add one before running")


# Limits that may only grow with explicit human approval (`--approve`).
GUARDED_LIMITS = ("budget_usd", "max_experiments", "sandbox.timeout_s", "sandbox.memory",
                  "sandbox.cpus", "sandbox.pids", "sandbox.network")


def guarded_values(cfg: Config) -> dict:
    return {
        "budget_usd": cfg.budget_usd, "max_experiments": cfg.max_experiments,
        "sandbox.timeout_s": cfg.sandbox.timeout_s, "sandbox.memory": _mem_bytes(cfg.sandbox.memory),
        "sandbox.cpus": cfg.sandbox.cpus, "sandbox.pids": cfg.sandbox.pids,
        "sandbox.network": 0 if cfg.sandbox.network == "none" else 1,
    }


def expansions(old: dict | None, new: dict) -> list[tuple[str, object, object]]:
    """Guarded limits that `new` raises relative to `old` (or that start above the safe default)."""
    if old is None:
        return [("sandbox.network", 0, 1)] if new["sandbox.network"] else []
    return [(k, old[k], new[k]) for k in GUARDED_LIMITS if new[k] > old[k]]


def _mem_bytes(s: str) -> int:
    s = str(s).strip().lower()
    mult = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}
    return int(float(s[:-1]) * mult[s[-1]]) if s[-1] in mult else int(s)
