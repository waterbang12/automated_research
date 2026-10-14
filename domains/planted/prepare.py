# Trusted data preparation: runs once per run inside the sandbox image.
# Reads the raw data from /raw (domains/planted/data, read-only), writes agent-visible inputs to
# /public and held-out targets to /private (never mounted for experiments).
import json

import numpy as np
import pandas as pd

TARGET = 'y'
TASK = 'regression'

df = pd.read_csv("/raw/data.csv")
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
rng = np.random.default_rng(20261014)          # fixed split: every run of this domain sees the same data
idx = rng.permutation(len(df))
n_train, n_val = int(0.6 * len(df)), int(0.2 * len(df))
parts = {"train": idx[:n_train], "val": idx[n_train:n_train + n_val], "test": idx[n_train + n_val:]}

df.iloc[parts["train"]].to_csv("/public/train.csv", index=False)
features = df.drop(columns=[TARGET])
for split in ("val", "test"):
    features.iloc[parts[split]].to_csv(f"/public/{split}_X.csv", index=False)
    df[TARGET].iloc[parts[split]].to_csv(f"/private/{split}_y.csv", index=False, header=False)
if TASK == "classification":
    with open("/private/labels.json", "w") as f:
        json.dump(sorted(df[TARGET].astype(str).unique().tolist()), f)
print({k: len(v) for k, v in parts.items()})
