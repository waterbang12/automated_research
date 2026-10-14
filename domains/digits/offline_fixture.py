"""Deterministic, credential-free stand-in for the LLM roles on the digits domain.

It is NOT an AI: it searches a fixed grid of sklearn models. It does read the
recorded experiment history the real prompts are built from (best result so
far, failures, what is pending), so the controller's feedback loop, lineage,
execution and scoring are exercised for real. With an API key configured, the
real LLM replaces all three functions.
"""
import json

GRID = {
    "svc_rbf": {"C": [1, 3, 10, 30, 100], "gamma": ["scale", 0.0005, 0.001, 0.002]},
    "knn": {"n_neighbors": [1, 3, 5, 7, 9], "weights": ["distance", "uniform"]},
    "mlp": {"hidden": ["128", "256", "256-128"], "alpha": [0.001, 0.0001, 0.01]},
    "extra_trees": {"n_estimators": [300, 100, 600], "max_features": ["sqrt", 0.3]},
    "random_forest": {"n_estimators": [300, 100, 600], "max_features": ["sqrt", 0.3]},
    "hgb": {"learning_rate": [0.1, 0.05, 0.2], "max_iter": [200, 100, 400]},
    "logreg": {"C": [1, 0.1, 10, 100]},
}
# Starting point per family: index into each grid list.
START = {"svc_rbf": {"C": 2, "gamma": 0}}


def _spec(method: str):
    marker = "SPEC "
    if marker not in method:
        return None
    try:
        return json.loads(method.split(marker, 1)[1].strip().splitlines()[0])
    except (json.JSONDecodeError, IndexError):
        return None


def _key(spec) -> str:
    return json.dumps(spec, sort_keys=True)


def _default(family):
    return {"family": family, "params": {k: v[START.get(family, {}).get(k, 0)] for k, v in GRID[family].items()}}


def _describe(spec) -> str:
    if spec["family"] == "vote":
        return "soft-vote ensemble of " + " + ".join(_describe(m) for m in spec["params"]["members"])
    return spec["family"] + "(" + ", ".join(f"{k}={v}" for k, v in spec["params"].items()) + ")"


def _hyp(spec, rationale, expected, priority, parents):
    return {"title": _describe(spec)[:80], "rationale": rationale,
            "method": f"Train {_describe(spec)} on standardised pixels.\nSPEC {json.dumps(spec)}",
            "expected_metric": round(expected, 4), "priority": round(priority, 3),
            "parent_experiment_ids": parents,
            "falsified_if": f"validation accuracy below {expected - 0.01:.4f}"}


def researcher(ctx):
    n = ctx["n"]
    base = ctx["baseline"]
    seen = {_key(s) for s in (_spec(e["method"] or "") for e in ctx["experiments"] + ctx["pending"]) if s}
    done = [(e, _spec(e["method"] or "")) for e in ctx["experiments"] if e["status"] == "completed"]
    done = [(e, s) for e, s in done if s]
    out = []

    def add(spec, *a):
        if _key(spec) not in seen and len(out) < n:
            seen.add(_key(spec))
            out.append(_hyp(spec, *a))

    single = [(e, s) for e, s in done if s["family"] in GRID]
    if single:
        # Exploit: one-step neighbours of the best single-model configuration so far.
        best, bspec = max(single, key=lambda es: es[0]["val_score"])
        for param, values in GRID[bspec["family"]].items():
            cur = values.index(bspec["params"][param]) if bspec["params"][param] in values else 0
            for j in (cur + 1, cur - 1):
                if 0 <= j < len(values):
                    nb = {"family": bspec["family"], "params": {**bspec["params"], param: values[j]}}
                    add(nb, f"Experiment #{best['id']} ({best['title']}) is the best so far at "
                            f"{best['val_score']:.4f}; moving {param} {bspec['params'][param]} -> {values[j]} "
                            f"tests whether it sits at a local optimum.",
                        best["val_score"] + 0.002, 0.9 - 0.05 * len(out), [best["id"]])
        # Combine: soft-vote the best configs of the two strongest families above baseline.
        by_family = {}
        for e, s in done:
            if s["family"] in GRID and e["improvement"] > 0:
                if s["family"] not in by_family or e["val_score"] > by_family[s["family"]][0]["val_score"]:
                    by_family[s["family"]] = (e, s)
        top = sorted(by_family.values(), key=lambda es: -es[0]["val_score"])[:2]
        if len(top) == 2:
            add({"family": "vote", "params": {"members": [top[0][1], top[1][1]]}},
                f"Experiments #{top[0][0]['id']} and #{top[1][0]['id']} are the strongest distinct families; "
                f"their errors may be partly independent.",
                top[0][0]["val_score"] + 0.003, 0.85, [top[0][0]["id"], top[1][0]["id"]])
    # Explore: families never tried, skipping ones that already lost to the baseline.
    losing = {s["family"] for e, s in done if e["improvement"] <= 0}
    tried = {s["family"] for e, s in done}
    for fam in GRID:
        if fam not in tried and fam not in losing:
            add(_default(fam), f"Family {fam} is untested; baseline is {base['val_score']:.4f}.",
                base["val_score"] + 0.01, 0.6 if done else 0.8 - 0.05 * len(out), [base["id"]])
    best = max(done, key=lambda es: es[0]["val_score"])[0] if done else None
    skipped = [f for f in GRID if f not in tried and f not in losing and
               not any(_spec(h["method"])["family"] == f for h in out)][:2]
    decision = {
        "current_belief": (f"Best so far is #{best['id']} at {best['val_score']:.4f} vs baseline {base['val_score']:.4f}."
                           if best else f"Only the baseline has been measured ({base['val_score']:.4f})."),
        "evidence_used": [best["id"]] if best else [base["id"]],
        "alternatives_considered": [{"idea": f"untested family {f}", "why_not_now": "lower priority than the "
                                     "ideas proposed; the grid search reaches it in a later round"} for f in skipped],
        "assumptions": ["the fixed validation split ranks models the same way the test split would"],
        "uncertain": ([f"whether #{best['id']} is a local optimum of its grid"] if best else []),
        "why_these": "Exploit the best configuration's grid neighbours, combine the strongest families, "
                     "then explore untested families (deterministic grid search, not AI).",
    }
    return {"decision": decision, "hypotheses": out}


def _model_code(spec) -> str:
    f, p = spec["family"], spec["params"]
    if f == "vote":
        members = ", ".join(f"('m{i}', {_model_code(m)})" for i, m in enumerate(p["members"]))
        return f"VotingClassifier([{members}], voting='soft')"
    if f == "svc_rbf":
        g = repr(p["gamma"])
        return f"make_pipeline(StandardScaler(), SVC(C={p['C']}, gamma={g}, probability=True, random_state=SEED))"
    if f == "knn":
        return f"make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors={p['n_neighbors']}, weights={p['weights']!r}))"
    if f == "mlp":
        hidden = tuple(int(h) for h in str(p["hidden"]).split("-"))
        return (f"make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes={hidden}, alpha={p['alpha']}, "
                f"max_iter=600, random_state=SEED))")
    if f in ("extra_trees", "random_forest"):
        cls = "ExtraTreesClassifier" if f == "extra_trees" else "RandomForestClassifier"
        return f"{cls}(n_estimators={p['n_estimators']}, max_features={p['max_features']!r}, random_state=SEED, n_jobs=1)"
    if f == "hgb":
        return f"HistGradientBoostingClassifier(learning_rate={p['learning_rate']}, max_iter={p['max_iter']}, random_state=SEED)"
    if f == "logreg":
        return f"make_pipeline(StandardScaler(), LogisticRegression(C={p['C']}, max_iter=3000, random_state=SEED))"
    raise ValueError(f"unknown family {f}")


TEMPLATE = '''import os
import numpy as np
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = int(os.environ.get("SEED", "0"))
train = np.loadtxt("/data/train.csv", delimiter=",", skiprows=1)
X, y = train[:, :-1], train[:, -1].astype(int)
X_val = np.loadtxt("/data/val_X.csv", delimiter=",", skiprows=1)
X_test = np.loadtxt("/data/test_X.csv", delimiter=",", skiprows=1)

model = {model}
model.fit(X, y)
print("train acc", model.score(X, y))
np.savetxt("/out/val_pred.csv", model.predict(X_val), fmt="%d")
np.savetxt("/out/test_pred.csv", model.predict(X_test), fmt="%d")
'''


def coder(ctx):
    spec = _spec(ctx["hypothesis"]["method"])
    if spec is None:
        raise ValueError("offline coder only understands hypotheses produced by the offline researcher")
    return {"code": TEMPLATE.format(model=_model_code(spec)),
            "explanation": f"Template implementation of {_describe(spec)}.", "requested_permissions": []}


def analyst(ctx):
    imp, need = ctx["improvement"], ctx["min_improvement"]
    verdict = "supported" if imp >= need else ("refuted" if imp < 0 else "inconclusive")
    return {"verdict": verdict,
            "analysis": f"Validation {ctx['val_score']:.4f} vs baseline {ctx['baseline_val']:.4f} "
                        f"(delta {imp:+.4f}, threshold {need}).",
            "lessons": [f"{ctx['hypothesis']['title']}: delta {imp:+.4f} over baseline"]}
