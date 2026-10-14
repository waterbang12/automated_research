# Trusted data preparation: runs once per run inside the sandbox image.
# Writes public inputs to /public and held-out labels to /private (never mounted for experiments).
import numpy as np
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
rng = np.random.default_rng(20261014)
idx = rng.permutation(len(y))
n_train, n_val = int(0.4 * len(y)), int(0.3 * len(y))
splits = {"train": idx[:n_train], "val": idx[n_train:n_train + n_val], "test": idx[n_train + n_val:]}

header = ",".join(f"f{i}" for i in range(X.shape[1]))
tr = splits["train"]
np.savetxt("/public/train.csv", np.column_stack([X[tr], y[tr]]), fmt="%d", delimiter=",",
           header=header + ",label", comments="")
for name in ("val", "test"):
    s = splits[name]
    np.savetxt(f"/public/{name}_X.csv", X[s], fmt="%d", delimiter=",", header=header, comments="")
    np.savetxt(f"/private/{name}_y.csv", y[s], fmt="%d")
print({k: len(v) for k, v in splits.items()})
