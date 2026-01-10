from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from src.features import get_feature_spec, split_X_y
from src.pipelines import make_xgb_pipeline

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "cs-training.csv" 
    
    artifacts_dir = project_root / "artifacts"
    (artifacts_dir / "models").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "metrics").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    spec = get_feature_spec(df)
    X, y = split_X_y(df, spec)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify = y, random_state = 42)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    pipe = make_xgb_pipeline(
        log_cols = spec.log_cols,
        other_cols = spec.other_cols,
        scale_pos_weight = scale_pos_weight
    )

    pipe.fit(X_train, y_train)

    train_pred = pipe.predict_proba(X_train)[:, 1]
    valid_pred = pipe.predict_proba(X_valid)[:, 1]

    train_auc = roc_auc_score(y_train, train_pred)
    valid_auc = roc_auc_score(y_valid, valid_pred)

    # save model
    model_path = artifacts_dir / "models" / "xgb_pipeline.joblib"
    joblib.dump(pipe, model_path)

    # save metrics
    metrics_path = artifacts_dir / "metrics" / "xgb_auc.csv"
    pd.DataFrame([{"train_auc": train_auc, "valid_auc": valid_auc}]).to_csv(metrics_path, index=False)

    print(f"Saved model -> {model_path}")
    print(f"Saved metrics -> {metrics_path}")
    print(f"AUC train={train_auc:.4f} valid={valid_auc:.4f}")


if __name__ == "__main__":
    main()