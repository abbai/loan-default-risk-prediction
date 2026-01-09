# src/modeling/pipelines.py
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def make_lr_pipeline(
    log_cols,
    other_cols,
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
    C=1.0,
):
    log_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ])

    other_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("log", log_pipe, log_cols),
            ("other", other_pipe, other_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            solver=solver,
            C=C,
        )),
    ])

    return pipe
