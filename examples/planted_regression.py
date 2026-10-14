"""Generate a regression dataset with planted structure and a known best-possible score.

Use it to *test the agent*: the target is built from a few nonlinear effects plus Gaussian
noise, so the irreducible error (the best RMSE any model can reach in expectation) is known:
NOISE_SD. A linear baseline cannot represent the structure, so you can measure how much of the
gap between the baseline and NOISE_SD the agent closes, how many experiments it needs, and
whether its recorded reasoning finds the real structure. The agent is never told the formula.

    python examples/planted_regression.py examples/planted.csv
"""
import csv
import math
import random
import sys

NOISE_SD = 0.5          # irreducible error: the best achievable RMSE in expectation
N = 3000


def target(x1, x2, x3, x4, group):
    effect = {"A": 0.0, "B": 1.5, "C": -1.0}[group]
    return (3.0 * math.sin(x1)               # smooth nonlinearity
            + 2.0 * x2 * x3                  # interaction
            + (2.0 if x4 > 0.5 else 0.0)     # threshold
            + effect)                        # categorical shift


def main(path: str) -> None:
    rng = random.Random(7)
    cols = ["x1", "x2", "x3", "x4", "group"] + [f"noise{i}" for i in range(1, 7)] + ["y"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for _ in range(N):
            x1, x2, x3 = rng.uniform(-3, 3), rng.gauss(0, 1), rng.gauss(0, 1)
            x4, group = rng.random(), rng.choice("ABC")
            noise = [round(rng.gauss(0, 1), 4) if rng.random() > 0.05 else "" for _ in range(6)]  # 5% missing
            y = target(x1, x2, x3, x4, group) + rng.gauss(0, NOISE_SD)
            w.writerow([round(x1, 4), round(x2, 4), round(x3, 4), round(x4, 4), group, *noise, round(y, 4)])
    print(f"wrote {N} rows to {path}; best achievable RMSE ~ {NOISE_SD}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/planted.csv")
