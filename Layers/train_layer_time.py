"""
Layer-level LATENCY prediction.

Trains 4 ML models (XGBoost, LightGBM, Random Forest, Neural Network)
to predict log1p(mean_ms) for individual convolution layers.

Evaluation: 80/20 random split + leave-one-model-out cross-validation.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import load_layer_data

SEED = 42
DATA_PATH = os.path.join(
    os.path.dirname(__file__), 'data', 'combined_layers.csv')
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')


# ============================================================================
# MODEL BUILDERS
# ============================================================================

def _build_preprocessor(numerical_cols, categorical_cols):
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False,
                              handle_unknown='infrequent_if_exist'),
         categorical_cols),
    ])


def build_xgb(numerical_cols, categorical_cols):
    return Pipeline([
        ('preprocessor', _build_preprocessor(numerical_cols, categorical_cols)),
        ('regressor', XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=SEED, tree_method='hist',
        )),
    ])


def build_lgbm(numerical_cols, categorical_cols):
    return Pipeline([
        ('preprocessor', _build_preprocessor(numerical_cols, categorical_cols)),
        ('regressor', LGBMRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=SEED, verbose=-1,
        )),
    ])


def build_rf(numerical_cols, categorical_cols):
    return Pipeline([
        ('preprocessor', _build_preprocessor(numerical_cols, categorical_cols)),
        ('regressor', RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_leaf=5,
            n_jobs=-1, random_state=SEED,
        )),
    ])


def build_nn(numerical_cols, categorical_cols):
    return Pipeline([
        ('preprocessor', _build_preprocessor(numerical_cols, categorical_cols)),
        ('regressor', MLPRegressor(
            hidden_layer_sizes=(512, 256, 128, 64, 32, 16, 8),
            activation='relu', solver='adam',
            learning_rate='adaptive', learning_rate_init=1e-3,
            max_iter=300, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=15,
            batch_size=1024, random_state=SEED, verbose=False,
        )),
    ])


MODEL_BUILDERS = {
    'xgb': build_xgb,
    'lgbm': build_lgbm,
    'rf': build_rf,
    'nn': build_nn,
}


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(y_true_log, y_pred_log, label: str):
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    r2_log = r2_score(y_true_log, y_pred_log)

    y_true_ms = np.expm1(y_true_log)
    y_pred_ms = np.expm1(y_pred_log)
    rmse_ms = np.sqrt(mean_squared_error(y_true_ms, y_pred_ms))
    mae_ms = mean_absolute_error(y_true_ms, y_pred_ms)
    r2_ms = r2_score(y_true_ms, y_pred_ms)

    mask = y_true_ms > 0
    mape = np.mean(np.abs(
        (y_true_ms[mask] - y_pred_ms[mask]) / y_true_ms[mask]
    )) * 100

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Log-scale   -- RMSE: {rmse_log:.4f}  MAE: {mae_log:.4f}"
          f"  R2: {r2_log:.4f}")
    print(f"  Original ms -- RMSE: {rmse_ms:.4f}  MAE: {mae_ms:.4f}"
          f"  R2: {r2_ms:.4f}  MAPE: {mape:.2f}%")

    return {
        'rmse_log': rmse_log, 'mae_log': mae_log, 'r2_log': r2_log,
        'rmse_ms': rmse_ms, 'mae_ms': mae_ms, 'r2_ms': r2_ms,
        'mape': mape,
    }


def print_feature_importances(pipe, numerical_cols, categorical_cols):
    """Print feature importances for tree-based models."""
    reg = pipe.named_steps['regressor']
    if not hasattr(reg, 'feature_importances_'):
        return

    cat_encoder = pipe.named_steps['preprocessor'].named_transformers_['cat']
    cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    feature_names = numerical_cols + cat_names
    importances = reg.feature_importances_

    print(f"\n{'Feature Importances':=^60}")
    for name, imp in sorted(
        zip(feature_names, importances), key=lambda x: -x[1]
    ):
        print(f"  {name:35s} {imp:.4f}")


# ============================================================================
# LEAVE-ONE-MODEL-OUT CV
# ============================================================================

def leave_one_model_out(
    df_full: pd.DataFrame, y_full: pd.Series,
    numerical_cols, categorical_cols,
):
    """Train on N-1 models, predict the held-out model."""
    models_in_data = df_full['model'].unique()
    print(f"\n{'=' * 60}")
    print("  LEAVE-ONE-MODEL-OUT CROSS-VALIDATION")
    print(f"{'=' * 60}")

    all_results = []
    for held_out in models_in_data:
        train_mask = df_full['model'] != held_out
        test_mask = df_full['model'] == held_out

        X_train = df_full[train_mask].drop(columns=['model'])
        y_train = y_full[train_mask]
        X_test = df_full[test_mask].drop(columns=['model'])
        y_test = y_full[test_mask]

        num_cols = [c for c in numerical_cols if c in X_train.columns]
        cat_cols = [c for c in categorical_cols if c in X_train.columns]

        pipe = build_xgb(num_cols, cat_cols)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        result = evaluate(y_test, y_pred,
                          f"Held out: {held_out} (XGBoost)")
        result['held_out'] = held_out
        all_results.append(result)

    return all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  LAYER-LEVEL LATENCY PREDICTION")
    print("=" * 60)

    X, y, _energy = load_layer_data(DATA_PATH, drop_model_col=False)

    model_col = X.pop('model') if 'model' in X.columns else None
    categorical_cols = ['algorithm', 'device']
    drop_cats = [c for c in categorical_cols if c in X.columns]
    for c in drop_cats:
        pass  # keep them for now
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    print(f"Features ({len(X.columns)}): {list(X.columns)}")
    print(f"Samples: {len(X):,}")
    print(f"Numerical: {numerical_cols}")
    print(f"Categorical: {categorical_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    results_summary = {}
    for name, builder in MODEL_BUILDERS.items():
        print(f"\n{'#' * 60}")
        print(f"  Training: {name.upper()}")
        print(f"{'#' * 60}")

        pipe = builder(numerical_cols, categorical_cols)
        pipe.fit(X_train, y_train)

        train_metrics = evaluate(
            y_train, pipe.predict(X_train), f"{name.upper()} -- Train")
        test_metrics = evaluate(
            y_test, pipe.predict(X_test), f"{name.upper()} -- Test")

        print_feature_importances(pipe, numerical_cols, categorical_cols)

        artifact_path = os.path.join(
            ARTIFACT_DIR, f'layer_time_{name}.joblib')
        joblib.dump(pipe, artifact_path)
        print(f"\n  Model saved to {artifact_path}")

        results_summary[name] = test_metrics

    # Summary table
    print(f"\n\n{'=' * 60}")
    print("  TEST SET COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Model':<8} {'RMSE(ms)':>10} {'MAE(ms)':>10} "
          f"{'R2':>8} {'MAPE%':>8}")
    print(f"  {'-' * 46}")
    for name, m in results_summary.items():
        print(f"  {name:<8} {m['rmse_ms']:>10.4f} {m['mae_ms']:>10.4f} "
              f"{m['r2_ms']:>8.4f} {m['mape']:>8.2f}")

    # Leave-one-model-out
    if model_col is not None:
        X_with_model = X.copy()
        X_with_model['model'] = model_col.values
        leave_one_model_out(X_with_model, y, numerical_cols, categorical_cols)


if __name__ == '__main__':
    main()
