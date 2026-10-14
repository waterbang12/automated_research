"""Trusted scorer. Runs on the host; reads candidate predictions and held-out labels."""
from pathlib import Path


class ScoreError(ValueError):
    pass


def _labels(path: Path, expected_rows: int | None = None) -> list[int]:
    if not path.exists():
        raise ScoreError(f"missing output file {path.name}")
    if path.stat().st_size > 1_000_000:
        raise ScoreError(f"{path.name} is implausibly large")
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").split()):
        try:
            v = int(float(line))
        except ValueError:
            raise ScoreError(f"{path.name} line {i + 1}: not a number: {line[:40]!r}")
        if not 0 <= v <= 9 or float(line) != v:
            raise ScoreError(f"{path.name} line {i + 1}: invalid label {line[:40]!r}")
        out.append(v)
    if expected_rows is not None and len(out) != expected_rows:
        raise ScoreError(f"{path.name}: expected {expected_rows} rows, got {len(out)}")
    return out


def accuracy(pred: list[int], truth: list[int]) -> float:
    return sum(p == t for p, t in zip(pred, truth)) / len(truth)


def score(out_dir: Path, private_dir: Path) -> dict:
    """Returns {"val": float, "test": float}. `val` is fed back to agents; `test` never is."""
    result = {}
    for split in ("val", "test"):
        truth = _labels(private_dir / f"{split}_y.csv")
        pred = _labels(out_dir / f"{split}_pred.csv", expected_rows=len(truth))
        result[split] = accuracy(pred, truth)
    return result
