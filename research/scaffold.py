"""Formulating a research problem: scaffold a domain from a CSV, and check a domain without any LLM.

`new-domain` writes a complete, editable tabular domain (classification -> accuracy,
regression -> RMSE) plus a run config. `check-domain` then proves the domain is sound before
an agent ever sees it: data preparation runs in the sandbox, labels end up only in the
private directory, the baseline runs and scores, scoring is deterministic, and the scorer
rejects missing, short and malformed outputs.
"""
from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path

from .domain import DOMAINS_DIR, Domain

ROOT = Path(__file__).resolve().parent.parent

DOMAIN_TOML = '''name = "{name}"
objective = "{objective}"
metric = "{metric}"
higher_is_better = {higher}
# A hypothesis "succeeds" when it beats the baseline's validation {metric} by at least this much.
# {threshold_note}
min_improvement = {threshold}

contract = """
You write ONE self-contained Python 3.12 program. It runs as `python /code/experiment.py`
inside a sandbox with NO network, a read-only filesystem except /out and /tmp, 1 CPU,
1 GB RAM and a hard wall-clock limit (see below). Installed libraries: numpy 2.1,
scipy 1.14, scikit-learn 1.5, pandas 2.2 and the standard library. Nothing else.

Task: {task} of the column `{target}` from the other columns of a tabular dataset.

Inputs (read-only CSV files with a header row):
  /data/train.csv  {n_train_note}all feature columns plus the target column `{target}`
  /data/val_X.csv  the same feature columns, no target
  /data/test_X.csv the same feature columns, no target
Feature columns: {columns}
{column_notes}
Required outputs:
  /out/val_pred.csv   one prediction per line, no header, same row order as val_X.csv
  /out/test_pred.csv  one prediction per line, no header, same row order as test_X.csv
{output_rule}

Rules:
  - Seed all randomness from the integer environment variable SEED.
  - Train only on /data/train.csv (you may split it further for model selection).
  - Do not try to read or infer validation/test targets; they are not available.
  - Handle missing values and non-numeric columns yourself.
  - Print short progress lines to stdout; keep total runtime well under the limit.
  - Exit with status 0 on success.
"""
'''

PREPARE = '''# Trusted data preparation: runs once per run inside the sandbox image.
# Reads the raw data from /raw (domains/{name}/data, read-only), writes agent-visible inputs to
# /public and held-out targets to /private (never mounted for experiments).
import json

import numpy as np
import pandas as pd

TARGET = {target!r}
TASK = {task!r}

df = pd.read_csv("/raw/{csv_name}")
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
rng = np.random.default_rng(20261014)          # fixed split: every run of this domain sees the same data
idx = rng.permutation(len(df))
n_train, n_val = int(0.6 * len(df)), int(0.2 * len(df))
parts = {{"train": idx[:n_train], "val": idx[n_train:n_train + n_val], "test": idx[n_train + n_val:]}}

df.iloc[parts["train"]].to_csv("/public/train.csv", index=False)
features = df.drop(columns=[TARGET])
for split in ("val", "test"):
    features.iloc[parts[split]].to_csv(f"/public/{{split}}_X.csv", index=False)
    df[TARGET].iloc[parts[split]].to_csv(f"/private/{{split}}_y.csv", index=False, header=False)
if TASK == "classification":
    with open("/private/labels.json", "w") as f:
        json.dump(sorted(df[TARGET].astype(str).unique().tolist()), f)
print({{k: len(v) for k, v in parts.items()}})
'''

SCORER = '''"""Trusted scorer. Runs on the host; reads candidate predictions and held-out targets."""
import json
import math
from pathlib import Path

TASK = {task!r}


class ScoreError(ValueError):
    pass


def _lines(path: Path, expected_rows: int | None = None) -> list[str]:
    if not path.exists():
        raise ScoreError(f"missing output file {{path.name}}")
    if path.stat().st_size > 50_000_000:
        raise ScoreError(f"{{path.name}} is implausibly large")
    rows = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    while rows and rows[-1] == "":
        rows.pop()
    if expected_rows is not None and len(rows) != expected_rows:
        raise ScoreError(f"{{path.name}}: expected {{expected_rows}} rows, got {{len(rows)}}")
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
        raise ScoreError(f"{{name}} line {{i + 1}}: not a number: {{v[:40]!r}}")
    if not math.isfinite(x):
        raise ScoreError(f"{{name}} line {{i + 1}}: not finite: {{v[:40]!r}}")
    return x


def score(out_dir: Path, private_dir: Path) -> dict:
    """Returns {{"val": float, "test": float}}. `val` is fed back to agents; `test` never is."""
    result = {{}}
    labels = None
    if TASK == "classification":
        labels = {{_label(x) for x in json.loads((private_dir / "labels.json").read_text(encoding="utf-8"))}}
    for split in ("val", "test"):
        truth = _lines(private_dir / f"{{split}}_y.csv")
        pred = _lines(out_dir / f"{{split}}_pred.csv", expected_rows=len(truth))
        name = f"{{split}}_pred.csv"
        if TASK == "classification":
            p = [_label(x) for x in pred]
            for i, x in enumerate(p):
                if x not in labels:
                    raise ScoreError(f"{{name}} line {{i + 1}}: unknown label {{pred[i][:40]!r}}")
            result[split] = sum(a == _label(b) for a, b in zip(p, truth)) / len(truth)
        else:
            p = [_number(x, name, i) for i, x in enumerate(pred)]
            t = [float(x) for x in truth]
            result[split] = math.sqrt(sum((a - b) ** 2 for a, b in zip(p, t)) / len(t))
    return result
'''

BASELINE = '''# Baseline: {baseline_desc}
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import {model_import}
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = int(os.environ.get("SEED", "0"))
TARGET = {target!r}

train = pd.read_csv("/data/train.csv")
y = train.pop(TARGET){y_cast}
numeric = train.select_dtypes("number").columns.tolist()
categorical = [c for c in train.columns if c not in numeric]
pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(), StandardScaler()), numeric),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical),
])
model = make_pipeline(pre, {model_ctor})
model.fit(train, y)
print("fitted on", len(train), "rows;", len(numeric), "numeric and", len(categorical), "categorical features")
for split in ("val", "test"):
    X = pd.read_csv(f"/data/{{split}}_X.csv")
    pd.Series(model.predict(X)).to_csv(f"/out/{{split}}_pred.csv", index=False, header=False)
'''


def _sniff(csv_path: Path, target: str) -> tuple[list[str], dict[str, str], int]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for _, r in zip(range(2000), reader)]
    if target not in header:
        raise SystemExit(f"target column {target!r} not in CSV header: {header}")
    kinds = {}
    for j, col in enumerate(header):
        vals = [r[j] for r in rows if j < len(r) and r[j].strip() != ""]
        try:
            [float(v) for v in vals]
            kinds[col] = "numeric"
        except ValueError:
            kinds[col] = "text/categorical"
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        n = sum(1 for _ in f) - 1
    return header, kinds, n


def new_domain(name: str, csv_path: Path, target: str, task: str | None, objective: str | None,
               domains_dir: Path = DOMAINS_DIR, configs_dir: Path = ROOT / "configs") -> list[Path]:
    dest = domains_dir / name
    if dest.exists():
        raise SystemExit(f"{dest} already exists; pick another name or edit it directly")
    header, kinds, n = _sniff(csv_path, target)
    if task is None:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            ys = {r[target] for r in csv.DictReader(f)}
        task = "regression" if kinds[target] == "numeric" and len(ys) > 20 else "classification"
    if task not in ("classification", "regression"):
        raise SystemExit("--task must be classification or regression")
    features = [c for c in header if c != target]
    cls = task == "classification"
    metric = "accuracy" if cls else "rmse"
    objective = objective or (f"Maximise validation accuracy predicting `{target}`." if cls
                              else f"Minimise validation RMSE predicting `{target}`.")
    (dest / "data").mkdir(parents=True)
    shutil.copy(csv_path, dest / "data" / "data.csv")
    text_cols = [c for c in features if kinds[c] != "numeric"]
    files = {
        "domain.toml": DOMAIN_TOML.format(
            name=name, objective=objective.replace('"', "'"), metric=metric, higher="true" if cls else "false",
            threshold=0.005 if cls else 0.0,
            threshold_note=("0.005 = half a percentage point of accuracy." if cls else
                            "Set this to the smallest RMSE drop that matters for your problem (0 = any drop)."),
            task="classification" if cls else "regression", target=target,
            n_train_note=f"(~{int(0.6 * n)} rows) ",
            columns=", ".join(features),
            column_notes=(f"Non-numeric columns: {', '.join(text_cols)}\n" if text_cols else ""),
            output_rule=("Each prediction must be one of the class labels that appear in train.csv's `"
                         + target + "` column." if cls else "Each prediction must be a finite number.")),
        "prepare.py": PREPARE.format(name=name, target=target, task=task, csv_name="data.csv"),
        "scorer.py": SCORER.format(task=task),
        "baseline.py": BASELINE.format(
            target=target,
            baseline_desc=("imputed, standardised / one-hot features + logistic regression" if cls else
                           "imputed, standardised / one-hot features + ridge regression"),
            model_import="LogisticRegression" if cls else "Ridge",
            model_ctor="LogisticRegression(max_iter=2000, random_state=SEED)" if cls else "Ridge()",
            y_cast=".astype(str)" if cls else ""),
    }
    written = []
    for fname, text in files.items():
        (dest / fname).write_text(text, encoding="utf-8")
        written.append(dest / fname)
    cfg = configs_dir / f"{name}.toml"
    if not cfg.exists():
        base = (configs_dir / "digits-claude-cli.toml").read_text(encoding="utf-8")
        base = base.replace('name = "digits-claude-cli"', f'name = "{name}"').replace(
            'domain = "digits"', f'domain = "{name}"')
        cfg.write_text(base, encoding="utf-8")
        written.append(cfg)
    return written


# --- checking a domain ----------------------------------------------------------
async def check_domain(name: str, executor, domains_dir: Path = DOMAINS_DIR) -> list[tuple[str, bool, str]]:
    """Run the domain end to end without an LLM. Returns (check, passed, detail) rows."""
    from .controller import prepare_mounts
    out: list[tuple[str, bool, str]] = []
    add = lambda label, ok, detail="": out.append((label, bool(ok), detail))
    try:
        dom = Domain(name, domains_dir)
    except Exception as e:
        add("domain.toml loads", False, str(e))
        return out
    add("domain.toml loads", True, f"metric {dom.metric} ({'higher' if dom.higher_is_better else 'lower'} is better), "
                                   f"min_improvement {dom.min_improvement}")
    for f in ("prepare.py", "scorer.py", "baseline.py"):
        add(f"{f} present", (dom.dir / f).exists())
    if not all(ok for _, ok, _ in out):
        return out
    scorer = dom.scorer()
    add("scorer defines score() and ScoreError", hasattr(scorer, "score") and hasattr(scorer, "ScoreError"))
    work = Path(tempfile.mkdtemp(prefix=f"check-{name}-"))
    try:
        await executor.preflight()
        pub, priv, code = work / "public", work / "private", work / "prepare"
        for d in (pub, priv, code):
            d.mkdir()
        (code / "prepare.py").write_text(dom.prepare_code(), encoding="utf-8")
        res = await executor.run(label="check-prepare", argv=["python", "/code/prepare.py"], env={},
                                 mounts=prepare_mounts(dom, code, pub, priv))
        add("prepare.py runs in the sandbox", res.ok, (res.stdout or res.stderr).strip()[-300:])
        if not res.ok:
            return out
        need_pub = {"train.csv", "val_X.csv", "test_X.csv"}
        have_pub = {p.name for p in pub.iterdir()}
        add("public inputs written", need_pub <= have_pub, ", ".join(sorted(have_pub)))
        add("held-out targets written to private only",
            {"val_y.csv", "test_y.csv"} <= {p.name for p in priv.iterdir()} and not any("_y" in n for n in have_pub),
            ", ".join(sorted(p.name for p in priv.iterdir())))
        bdir = work / "baseline"
        (bdir / "code").mkdir(parents=True)
        (bdir / "out").mkdir()
        (bdir / "code" / "experiment.py").write_text(dom.baseline_code(), encoding="utf-8")
        res = await executor.run(label="check-baseline", argv=["python", "/code/experiment.py"],
                                 env={"SEED": "1", "PYTHONHASHSEED": "0"},
                                 mounts=[(bdir / "code", "/code", True), (pub, "/data", True), (bdir / "out", "/out", False)])
        add("baseline runs in the sandbox", res.ok, f"{res.duration_s:.1f}s; " + (res.stdout or res.stderr).strip()[-200:])
        if not res.ok:
            return out
        try:
            s1 = scorer.score(bdir / "out", priv)
            s2 = scorer.score(bdir / "out", priv)
            add("baseline scores", True, f"val {dom.metric} {s1['val']:.4f}, test {s1['test']:.4f}")
            add("scoring is deterministic", s1 == s2)
        except Exception as e:
            add("baseline scores", False, f"{type(e).__name__}: {e}")
            return out

        def rejects(label: str, mutate) -> None:
            bad = work / f"bad-{len(out)}"
            shutil.copytree(bdir / "out", bad)
            mutate(bad)
            try:
                scorer.score(bad, priv)
                add(f"scorer rejects {label}", False, "accepted it")
            except scorer.ScoreError as e:
                add(f"scorer rejects {label}", True, str(e)[:120])
            except Exception as e:
                add(f"scorer rejects {label}", False, f"raised {type(e).__name__} instead of ScoreError: {e}")

        rejects("a missing output file", lambda d: (d / "val_pred.csv").unlink())
        rejects("a short output file", lambda d: (d / "val_pred.csv").write_text(
            "\n".join((d / "val_pred.csv").read_text().splitlines()[:-1]) + "\n"))
        rejects("malformed predictions", lambda d: (d / "val_pred.csv").write_text(
            "\n".join("not-a-prediction" for _ in (d / "val_pred.csv").read_text().splitlines()) + "\n"))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out
