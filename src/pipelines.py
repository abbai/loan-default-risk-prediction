# src/modeling/pipelines.py
import numpy as np
from xgboost import XGBClassifier
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

def make_xgb_pipeline(
    log_cols,
    other_cols,
    *,
    # best params
    n_estimators=900,
    learning_rate=0.02,
    max_depth=3,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.6,
    gamma=0.1,
    reg_lambda=2.0,
    reg_alpha=0.0,
    scale_pos_weight=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
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

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method=tree_method,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    return pipe