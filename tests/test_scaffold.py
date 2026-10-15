"""Problem formulation: scaffolding a domain from a CSV, the generated scorer, and the no-LLM domain check."""
import asyncio
import csv
import random

import pytest

from research.config import SandboxConfig
from research.domain import Domain
from research.executor import DockerExecutor, ExecutorUnavailable
from research.scaffold import check_domain, new_domain


def write_csv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def dirs(tmp_path):
    domains, configs = tmp_path / "domains", tmp_path / "configs"
    domains.mkdir()
    configs.mkdir()
    (configs / "digits-claude-cli.toml").write_text('[run]\nname = "digits-claude-cli"\ndomain = "digits"\n')
    return domains, configs


def classification_csv(path, n=300):
    rng = random.Random(1)
    rows = []
    for _ in range(n):
        a, b, color = rng.gauss(0, 1), rng.gauss(0, 1), rng.choice(["red", "blue"])
        rows.append([round(a, 3), round(b, 3) if rng.random() > 0.1 else "", color, int(a + (color == "red") > 0.5)])
    write_csv(path, rows, ["a", "b", "color", "label"])


def test_new_domain_writes_a_complete_editable_domain(tmp_path, dirs):
    domains, configs = dirs
    classification_csv(tmp_path / "d.csv")
    written = new_domain("toy", tmp_path / "d.csv", "label", None, None, domains, configs)
    assert {p.name for p in written} == {"domain.toml", "prepare.py", "scorer.py", "baseline.py", "toy.toml"}
    d = Domain("toy", domains)
    assert d.metric == "accuracy" and d.higher_is_better and d.min_improvement == 0.005   # inferred: few labels
    assert "a, b, color" in d.contract and "Non-numeric columns: color" in d.contract
    assert (domains / "toy" / "data" / "data.csv").exists()
    assert "/raw/data.csv" in d.prepare_code() and d.baseline_code().startswith("# Baseline:")
    cfg = (configs / "toy.toml").read_text()
    assert 'name = "toy"' in cfg and 'domain = "toy"' in cfg
    for f in ("prepare.py", "scorer.py", "baseline.py"):
        compile((domains / "toy" / f).read_text(), f, "exec")
    with pytest.raises(SystemExit):
        new_domain("toy", tmp_path / "d.csv", "label", None, None, domains, configs)   # never overwrites
    with pytest.raises(SystemExit):
        new_domain("other", tmp_path / "d.csv", "nope", None, None, domains, configs)  # unknown target


def test_generated_scorers(tmp_path, dirs):
    domains, configs = dirs
    classification_csv(tmp_path / "c.csv")
    write_csv(tmp_path / "r.csv", [[i, i * 0.5 + 0.1] for i in range(100)], ["x", "y"])
    new_domain("cls", tmp_path / "c.csv", "label", None, None, domains, configs)
    new_domain("reg", tmp_path / "r.csv", "y", None, None, domains, configs)
    assert Domain("reg", domains).metric == "rmse" and not Domain("reg", domains).higher_is_better
    priv, out = tmp_path / "priv", tmp_path / "out"
    priv.mkdir()
    out.mkdir()
    # classification: 1, 1.0 and "1" are the same label; unknown labels are rejected
    sc = Domain("cls", domains).scorer()
    (priv / "labels.json").write_text('["0", "1"]')
    for s in ("val", "test"):
        (priv / f"{s}_y.csv").write_text("1\n0\n1\n0\n")
        (out / f"{s}_pred.csv").write_text("1.0\n0\n0\n0\n")
    assert sc.score(out, priv) == {"val": 0.75, "test": 0.75}
    (out / "val_pred.csv").write_text("1\n0\n2\n0\n")
    with pytest.raises(sc.ScoreError, match="unknown label"):
        sc.score(out, priv)
    # regression: RMSE, non-finite rejected
    sr = Domain("reg", domains).scorer()
    for s in ("val", "test"):
        (priv / f"{s}_y.csv").write_text("1\n2\n3\n")
        (out / f"{s}_pred.csv").write_text("1\n2\n5\n")
    assert sr.score(out, priv)["val"] == pytest.approx((4 / 3) ** 0.5)
    (out / "val_pred.csv").write_text("1\nnan\n3\n")
    with pytest.raises(sr.ScoreError, match="not finite"):
        sr.score(out, priv)


def test_check_domain_end_to_end_in_docker(tmp_path, dirs):
    ex = DockerExecutor(SandboxConfig(timeout_s=120))
    try:
        asyncio.run(ex.preflight())
    except ExecutorUnavailable as e:
        pytest.skip(str(e))
    domains, configs = dirs
    classification_csv(tmp_path / "d.csv")
    new_domain("toy", tmp_path / "d.csv", "label", None, None, domains, configs)
    rows = asyncio.run(check_domain("toy", ex, domains))
    failed = [r for r in rows if not r[1]]
    assert not failed, failed
    assert any(label == "baseline scores" for label, _, _ in rows)
    # A scorer that accepts anything is caught.
    sc = domains / "toy" / "scorer.py"
    sc.write_text(sc.read_text().replace('raise ScoreError(f"{name} line {i + 1}: unknown label', 'pass  # (', 1)
                  .replace("{pred[i][:40]!r}\")", "", 1))
    rows = asyncio.run(check_domain("toy", ex, domains))
    assert ("scorer rejects malformed predictions", False) in [(label, ok) for label, ok, _ in rows]


def test_planted_problem_has_the_stated_noise_floor(tmp_path):
    """The agent evaluation relies on the planted data's best achievable RMSE being ~NOISE_SD."""
    import importlib.util
    import math
    from pathlib import Path
    gen_path = Path(__file__).resolve().parent.parent / "examples" / "planted_regression.py"
    spec = importlib.util.spec_from_file_location("planted_gen", gen_path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    out = tmp_path / "planted.csv"
    gen.main(str(out))
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == gen.N and {"x1", "x2", "x3", "x4", "group", "noise1", "y"} <= set(rows[0])
    resid = [float(r["y"]) - gen.target(float(r["x1"]), float(r["x2"]), float(r["x3"]), float(r["x4"]), r["group"])
             for r in rows]
    rmse_of_truth = math.sqrt(sum(e * e for e in resid) / len(resid))
    assert abs(rmse_of_truth - gen.NOISE_SD) < 0.03        # the true function scores ~0.5: the floor
