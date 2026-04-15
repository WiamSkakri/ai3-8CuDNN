"""
Unified evaluation script for all 4 prediction tasks.

Loads saved model artifacts and evaluates them on all four targets:
  1. Layer-level time    (mean_ms)
  2. Layer-level energy  (energy_per_layer_j)
  3. Model-level time    (mean)
  4. Model-level energy  (energy_per_inference_j)

Produces:
  - Per-task comparison tables (RMSE, MAE, R2, MAPE)
  - Per-model (architecture) breakdown
  - Per-algorithm breakdown
  - Per-device breakdown
  - Leave-one-model-out CV results
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import load_layer_data, load_overall_data, add_layer_features

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

ML_MODELS = ['xgb', 'lgbm', 'rf', 'nn']


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(y_true_log, y_pred_log, unit='ms'):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mask = y_true > 0

    return {
        'rmse_log': np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        'mae_log': mean_absolute_error(y_true_log, y_pred_log),
        'r2_log': r2_score(y_true_log, y_pred_log),
        f'rmse_{unit}': np.sqrt(mean_squared_error(y_true, y_pred)),
        f'mae_{unit}': mean_absolute_error(y_true, y_pred),
        f'r2_{unit}': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs(
            (y_true[mask] - y_pred[mask]) / y_true[mask]
        )) * 100 if mask.any() else 0,
    }


# ============================================================================
# BREAKDOWN BY GROUP
# ============================================================================

def breakdown_by_group(
    pipe, X: pd.DataFrame, y: pd.Series,
    group_col: str, group_values: np.ndarray,
    unit: str = 'ms',
):
    """Compute metrics per unique value of group_col."""
    rows = []
    y_pred = pipe.predict(X)
    for val in sorted(group_values.unique()):
        mask = group_values == val
        if mask.sum() == 0:
            continue
        m = compute_metrics(y.values[mask], y_pred[mask], unit=unit)
        m['group'] = val
        rows.append(m)
    return pd.DataFrame(rows)


# ============================================================================
# TASK EVALUATORS
# ============================================================================

def evaluate_layer_time():
    csv_path = os.path.join(DATA_DIR, 'combined_layers.csv')
    if not os.path.exists(csv_path):
        print("  SKIP: combined_layers.csv not found")
        return

    X, y, _energy = load_layer_data(csv_path, drop_model_col=False)
    raw_df = pd.read_csv(csv_path)
    model_names = raw_df['model'] if 'model' in raw_df.columns else None
    algorithm_vals = X['algorithm'] if 'algorithm' in X.columns else None
    device_vals = X['device'] if 'device' in X.columns else None

    if 'model' in X.columns:
        X = X.drop(columns=['model'])

    print(f"\n{'#' * 70}")
    print("  TASK 1: LAYER-LEVEL TIME PREDICTION")
    print(f"{'#' * 70}")

    _run_all_models('layer_time', X, y, 'ms',
                    model_names, algorithm_vals, device_vals)


def evaluate_layer_energy():
    csv_path = os.path.join(DATA_DIR, 'combined_layers.csv')
    if not os.path.exists(csv_path):
        print("  SKIP: combined_layers.csv not found")
        return

    df = pd.read_csv(csv_path)
    if 'energy_per_layer_j' not in df.columns:
        print("  SKIP: energy_per_layer_j column not found "
              "(requires v2 profiler data)")
        return

    model_names = df['model'] if 'model' in df.columns else None
    algorithm_vals = df['algorithm'] if 'algorithm' in df.columns else None
    device_vals = df['device'] if 'device' in df.columns else None

    y = np.log1p(df['energy_per_layer_j'])

    drop_cols = ['model', 'layer', 'mean_ms', 'std_ms', 'percentage',
                 'power_mean_w', 'power_std_w', 'power_samples',
                 'energy_per_layer_j', 'energy_attribution']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = add_layer_features(X)

    print(f"\n{'#' * 70}")
    print("  TASK 2: LAYER-LEVEL ENERGY PREDICTION")
    print(f"{'#' * 70}")

    _run_all_models('layer_energy', X, y, 'j',
                    model_names, algorithm_vals, device_vals)


def evaluate_model_time():
    csv_path = os.path.join(DATA_DIR, 'combined_overall.csv')
    if not os.path.exists(csv_path):
        print("  SKIP: combined_overall.csv not found")
        return

    X, y_time, _y_energy = load_overall_data(csv_path)
    raw_df = pd.read_csv(csv_path)
    model_names = raw_df['model'] if 'model' in raw_df.columns else None
    algorithm_vals = X['algorithm'] if 'algorithm' in X.columns else None
    device_vals = X['device'] if 'device' in X.columns else None

    if 'model' in X.columns:
        X = X.drop(columns=['model'])

    print(f"\n{'#' * 70}")
    print("  TASK 3: MODEL-LEVEL TIME PREDICTION")
    print(f"{'#' * 70}")

    _run_all_models('model_time', X, y_time, 'ms',
                    model_names, algorithm_vals, device_vals)


def evaluate_model_energy():
    csv_path = os.path.join(DATA_DIR, 'combined_overall.csv')
    if not os.path.exists(csv_path):
        print("  SKIP: combined_overall.csv not found")
        return

    X, _y_time, y_energy = load_overall_data(csv_path)
    if y_energy is None:
        print("  SKIP: energy_per_inference_j column not found")
        return

    raw_df = pd.read_csv(csv_path)
    model_names = raw_df['model'] if 'model' in raw_df.columns else None
    algorithm_vals = X['algorithm'] if 'algorithm' in X.columns else None
    device_vals = X['device'] if 'device' in X.columns else None

    if 'model' in X.columns:
        X = X.drop(columns=['model'])

    print(f"\n{'#' * 70}")
    print("  TASK 4: MODEL-LEVEL ENERGY PREDICTION")
    print(f"{'#' * 70}")

    _run_all_models('model_energy', X, y_energy, 'j',
                    model_names, algorithm_vals, device_vals)


# ============================================================================
# SHARED RUNNER
# ============================================================================

def _run_all_models(
    task_prefix: str,
    X: pd.DataFrame,
    y: pd.Series,
    unit: str,
    model_names,
    algorithm_vals,
    device_vals,
):
    summary_rows = []

    for ml_name in ML_MODELS:
        artifact_path = os.path.join(
            ARTIFACT_DIR, f'{task_prefix}_{ml_name}.joblib')
        if not os.path.exists(artifact_path):
            print(f"  {ml_name.upper()}: artifact not found, skipping")
            continue

        pipe = joblib.load(artifact_path)
        y_pred = pipe.predict(X)
        m = compute_metrics(y.values, y_pred, unit=unit)
        m['model'] = ml_name.upper()
        summary_rows.append(m)

        print(f"\n  {ml_name.upper()}: "
              f"RMSE({unit})={m[f'rmse_{unit}']:.4f}  "
              f"MAE({unit})={m[f'mae_{unit}']:.4f}  "
              f"R2={m[f'r2_{unit}']:.4f}  "
              f"MAPE={m['mape']:.2f}%")

        # Per-architecture breakdown
        if model_names is not None:
            bd = breakdown_by_group(
                pipe, X, y, 'model', model_names, unit)
            print(f"\n    Per-architecture ({ml_name.upper()}):")
            for _, row in bd.iterrows():
                print(f"      {row['group']:15s}  "
                      f"RMSE={row[f'rmse_{unit}']:.4f}  "
                      f"R2={row[f'r2_{unit}']:.4f}  "
                      f"MAPE={row['mape']:.2f}%")

        # Per-algorithm breakdown
        if algorithm_vals is not None:
            bd = breakdown_by_group(
                pipe, X, y, 'algorithm', algorithm_vals, unit)
            print(f"\n    Per-algorithm ({ml_name.upper()}):")
            for _, row in bd.iterrows():
                print(f"      {row['group']:25s}  "
                      f"RMSE={row[f'rmse_{unit}']:.4f}  "
                      f"R2={row[f'r2_{unit}']:.4f}  "
                      f"MAPE={row['mape']:.2f}%")

        # Per-device breakdown
        if device_vals is not None:
            bd = breakdown_by_group(
                pipe, X, y, 'device', device_vals, unit)
            print(f"\n    Per-device ({ml_name.upper()}):")
            for _, row in bd.iterrows():
                print(f"      {row['group']:10s}  "
                      f"RMSE={row[f'rmse_{unit}']:.4f}  "
                      f"R2={row[f'r2_{unit}']:.4f}  "
                      f"MAPE={row['mape']:.2f}%")

    if summary_rows:
        print(f"\n  {'=' * 60}")
        print(f"  COMPARISON TABLE: {task_prefix.replace('_', ' ').upper()}")
        print(f"  {'=' * 60}")
        print(f"  {'ML Model':<8} {'RMSE':>12} {'MAE':>12} "
              f"{'R2':>8} {'MAPE%':>8}")
        print(f"  {'-' * 50}")
        for row in summary_rows:
            print(f"  {row['model']:<8} "
                  f"{row[f'rmse_{unit}']:>12.4f} "
                  f"{row[f'mae_{unit}']:>12.4f} "
                  f"{row[f'r2_{unit}']:>8.4f} "
                  f"{row['mape']:>8.2f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  UNIFIED EVALUATION -- ALL 4 PREDICTION TASKS")
    print("=" * 70)

    evaluate_layer_time()
    evaluate_layer_energy()
    evaluate_model_time()
    evaluate_model_energy()

    print(f"\n\n{'=' * 70}")
    print("  EVALUATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
