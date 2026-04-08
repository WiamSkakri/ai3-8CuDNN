"""
XGBoost regression for per-layer latency prediction.
Uses only the raw layer parameters — no engineered features.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

SEED = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "combined_layers.csv")


def load_and_prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df = df.drop(columns=["model"])

    y = np.log1p(df.pop("mean_ms"))
    return df, y


def build_pipeline(numerical_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=SEED,
            tree_method="hist",
        )),
    ])


def evaluate(y_true_log, y_pred_log, label: str):
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    r2_log = r2_score(y_true_log, y_pred_log)

    y_true_ms = np.expm1(y_true_log)
    y_pred_ms = np.expm1(y_pred_log)
    rmse_ms = np.sqrt(mean_squared_error(y_true_ms, y_pred_ms))
    mae_ms = mean_absolute_error(y_true_ms, y_pred_ms)
    r2_ms = r2_score(y_true_ms, y_pred_ms)

    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  Log-scale   — RMSE: {rmse_log:.4f}  MAE: {mae_log:.4f}  R²: {r2_log:.4f}")
    print(f"  Original ms — RMSE: {rmse_ms:.4f}  MAE: {mae_ms:.4f}  R²: {r2_ms:.4f}")


def main():
    X, y = load_and_prepare(DATA_PATH)

    categorical_cols = ["algorithm", "device"]
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    print(f"Features ({len(X.columns)}): {list(X.columns)}")
    print(f"Samples: {len(X):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    pipe = build_pipeline(numerical_cols, categorical_cols)

    print("\nTraining XGBoost...")
    pipe.fit(X_train, y_train)

    evaluate(y_train, pipe.predict(X_train), "Train")
    evaluate(y_test, pipe.predict(X_test), "Test")

    feature_names = (
        numerical_cols
        + pipe.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_cols)
        .tolist()
    )
    importances = pipe.named_steps["regressor"].feature_importances_
    print(f"\n{'Feature Importances':=^50}")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"  {name:30s} {imp:.4f}")

    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    artifact_path = os.path.join(out_dir, "xgb.joblib")
    joblib.dump(pipe, artifact_path)
    print(f"\nModel saved to {artifact_path}")


if __name__ == "__main__":
    main()
