# src/features.py
import pandas as pd

TARGET_COL = "SeriousDlqin2yrs"


class FeatureSpec:
    """
    Simple container for feature definitions.
    No decorators, no typing magic — 100% reliable.
    """
    def __init__(self, target, log_cols, other_cols):
        self.target = target
        self.log_cols = log_cols
        self.other_cols = other_cols


def get_feature_spec(df):
    log_cols = [
        "MonthlyIncome",
        "DebtRatio",
        "RevolvingUtilizationOfUnsecuredLines",
    ]

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    other_cols = [c for c in feature_cols if c not in log_cols]

    return FeatureSpec(
        target=TARGET_COL,
        log_cols=log_cols,
        other_cols=other_cols,
    )


def split_X_y(df, spec):
    X = df.drop(columns=[spec.target])
    y = df[spec.target]
    return X, y
