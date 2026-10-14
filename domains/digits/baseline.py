# Baseline: standardised features + multinomial logistic regression.
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = int(os.environ.get("SEED", "0"))
train = np.loadtxt("/data/train.csv", delimiter=",", skiprows=1)
X, y = train[:, :-1], train[:, -1].astype(int)
X_val = np.loadtxt("/data/val_X.csv", delimiter=",", skiprows=1)
X_test = np.loadtxt("/data/test_X.csv", delimiter=",", skiprows=1)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
model.fit(X, y)
print("train acc", model.score(X, y))
np.savetxt("/out/val_pred.csv", model.predict(X_val), fmt="%d")
np.savetxt("/out/test_pred.csv", model.predict(X_test), fmt="%d")
