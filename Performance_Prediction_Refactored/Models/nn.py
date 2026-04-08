"""
Neural Network (MLPRegressor) for per-layer latency prediction.
7 hidden layers with tapering widths, ReLU activation.
Uses only the raw layer parameters — no engineered features.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
        ("regressor", MLPRegressor(
            hidden_layer_sizes=(512, 256, 128, 64, 32, 16, 8),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            batch_size=1024,
            random_state=SEED,
            verbose=True,
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

    print("\nTraining Neural Network (7-layer MLP)...")
    pipe.fit(X_train, y_train)

    evaluate(y_train, pipe.predict(X_train), "Train")
    evaluate(y_test, pipe.predict(X_test), "Test")

    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    artifact_path = os.path.join(out_dir, "nn.joblib")
    joblib.dump(pipe, artifact_path)
    print(f"\nModel saved to {artifact_path}")


if __name__ == "__main__":
    main()
