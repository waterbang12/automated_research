# Baseline: imputed, standardised / one-hot features + ridge regression
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = int(os.environ.get("SEED", "0"))
TARGET = 'y'

train = pd.read_csv("/data/train.csv")
y = train.pop(TARGET)
numeric = train.select_dtypes("number").columns.tolist()
categorical = [c for c in train.columns if c not in numeric]
pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(), StandardScaler()), numeric),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical),
])
model = make_pipeline(pre, Ridge())
model.fit(train, y)
print("fitted on", len(train), "rows;", len(numeric), "numeric and", len(categorical), "categorical features")
for split in ("val", "test"):
    X = pd.read_csv(f"/data/{split}_X.csv")
    pd.Series(model.predict(X)).to_csv(f"/out/{split}_pred.csv", index=False, header=False)
