"""A research domain: a directory with a task contract, data preparation, a trusted
scorer, and a baseline. Nothing in a domain directory is ever mounted into the
sandbox, and its files are hashed at run start and re-checked before each score.
"""
from __future__ import annotations

import hashlib
import importlib.util
import tomllib
from pathlib import Path

DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains"


class IntegrityError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Domain:
    def __init__(self, name: str, base: Path = DOMAINS_DIR):
        self.dir = (base / name).resolve()
        spec = tomllib.loads((self.dir / "domain.toml").read_text(encoding="utf-8"))
        self.name = spec["name"]
        self.objective = spec["objective"]
        self.contract = spec["contract"]
        self.metric = spec["metric"]
        self.higher_is_better = bool(spec.get("higher_is_better", True))
        self.min_improvement = float(spec.get("min_improvement", 0.0))
        self.spec = spec

    def _load(self, filename: str):
        path = self.dir / filename
        if not path.exists():
            return None
        mod_name = f"domain_{self.name}_{path.stem}"
        s = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
        return mod

    def scorer(self):
        return self._load("scorer.py")

    def offline_fixture(self):
        return self._load("offline_fixture.py")

    def baseline_code(self) -> str:
        return (self.dir / "baseline.py").read_text(encoding="utf-8")

    def prepare_code(self) -> str:
        return (self.dir / "prepare.py").read_text(encoding="utf-8")

    def improvement(self, score: float, baseline: float) -> float:
        """Signed improvement where positive is always better."""
        return score - baseline if self.higher_is_better else baseline - score

    # --- integrity -------------------------------------------------------
    PROTECTED = ("domain.toml", "scorer.py", "prepare.py", "baseline.py")

    def snapshot(self, private_dir: Path) -> dict[str, str]:
        """Hashes of everything that defines the evaluation: contract, scorer, data split, baseline, labels."""
        files = {f"domain/{n}": sha256_file(self.dir / n) for n in self.PROTECTED}
        files.update({f"private/{p.name}": sha256_file(p) for p in sorted(private_dir.glob("*")) if p.is_file()})
        return files

    def verify(self, private_dir: Path, expected: dict[str, str]) -> None:
        now = self.snapshot(private_dir)
        changed = sorted(k for k in set(now) | set(expected) if now.get(k) != expected.get(k))
        if changed:
            raise IntegrityError(f"evaluation assets changed since run start: {changed}")
