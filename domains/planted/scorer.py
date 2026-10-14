"""Trusted scorer. Runs on the host; reads candidate predictions and held-out targets."""
import json
import math
from pathlib import Path

TASK = 'regression'


class ScoreError(ValueError):
    pass


def _lines(path: Path, expected_rows: int | None = None) -> list[str]:
    if not path.exists():
        raise ScoreError(f"missing output file {path.name}")
    if path.stat().st_size > 50_000_000:
        raise ScoreError(f"{path.name} is implausibly large")
    rows = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    while rows and rows[-1] == "":
        rows.pop()
    if expected_rows is not None and len(rows) != expected_rows:
        raise ScoreError(f"{path.name}: expected {expected_rows} rows, got {len(rows)}")
    return rows


def _label(v: str) -> str:
    """Treat 1, 1.0 and "1" as the same label; leave text labels alone."""
    v = v.strip().strip('"')
    try:
        x = float(v)
        return str(int(x)) if x.is_integer() else repr(x)
    except ValueError:
        return v


def _number(v: str, name: str, i: int) -> float:
    try:
        x = float(v)
    except ValueError:
        raise ScoreError(f"{name} line {i + 1}: not a number: {v[:40]!r}")
    if not math.isfinite(x):
        raise ScoreError(f"{name} line {i + 1}: not finite: {v[:40]!r}")
    return x


def score(out_dir: Path, private_dir: Path) -> dict:
    """Returns {"val": float, "test": float}. `val` is fed back to agents; `test` never is."""
    result = {}
    labels = None
    if TASK == "classification":
        labels = {_label(x) for x in json.loads((private_dir / "labels.json").read_text(encoding="utf-8"))}
    for split in ("val", "test"):
        truth = _lines(private_dir / f"{split}_y.csv")
        pred = _lines(out_dir / f"{split}_pred.csv", expected_rows=len(truth))
        name = f"{split}_pred.csv"
        if TASK == "classification":
            p = [_label(x) for x in pred]
            for i, x in enumerate(p):
                if x not in labels:
                    raise ScoreError(f"{name} line {i + 1}: unknown label {pred[i][:40]!r}")
            result[split] = sum(a == _label(b) for a, b in zip(p, truth)) / len(truth)
        else:
            p = [_number(x, name, i) for i, x in enumerate(pred)]
            t = [float(x) for x in truth]
            result[split] = math.sqrt(sum((a - b) ** 2 for a, b in zip(p, t)) / len(t))
    return result
