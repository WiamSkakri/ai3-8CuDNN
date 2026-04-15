"""
Feature engineering for layer-level and model-level prediction.

Provides functions that load raw profiling CSVs, compute engineered features,
and return DataFrames ready for ML training.
"""

import math
import os
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd

from device_specs import DEVICE_SPECS, DEVICE_FEATURE_NAMES


# ============================================================================
# LAYER-LEVEL FEATURES
# ============================================================================

def compute_output_size(input_size, kernel_size, stride, padding):
    """Compute spatial output dimension for a conv layer."""
    return math.floor((input_size + 2 * padding - kernel_size) / stride) + 1


def add_layer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a layer-level DataFrame.

    Expected input columns:
        in_channels, out_channels, kernel_size, stride, padding, input_size

    Added columns:
        output_size, compute_flops, params, memory_bytes,
        flops_per_param, compute_intensity,
        + all device_specs numerical columns
    """
    df = df.copy()

    df['output_size'] = df.apply(
        lambda r: compute_output_size(
            r['input_size'], r['kernel_size'], r['stride'], r['padding']
        ), axis=1,
    )

    # Exact Conv2D FLOPs: 2 * Cin * Cout * K^2 * Hout * Wout
    df['compute_flops'] = (
        2.0
        * df['in_channels']
        * df['out_channels']
        * (df['kernel_size'] ** 2)
        * (df['output_size'] ** 2)
    )

    df['params'] = (
        df['in_channels'] * df['out_channels'] * (df['kernel_size'] ** 2)
    )

    # Memory footprint in bytes (float32):
    # input + output + weights
    df['memory_bytes'] = (
        (df['in_channels'] * (df['input_size'] ** 2))
        + (df['out_channels'] * (df['output_size'] ** 2))
        + df['params']
    ) * 4.0

    df['flops_per_param'] = df['compute_flops'] / df['params'].clip(lower=1)

    # Operational (arithmetic) intensity: FLOPs / bytes moved
    df['compute_intensity'] = (
        df['compute_flops'] / df['memory_bytes'].clip(lower=1)
    )

    df = _add_device_features(df)

    return df


# ============================================================================
# MODEL-LEVEL FEATURES
# ============================================================================

# Architecture-level features extracted statically from the model name.
# These are used so the predictor can generalise to unseen models if their
# architecture features are provided.

_MODEL_ARCH_FEATURES: Dict[str, Dict[str, float]] = {
    'vgg16': {
        'num_conv_layers': 13,
        'total_params': 138_357_544,
        'total_conv_params': 14_714_688,
        'depth': 16,
        'max_channels': 512,
        'has_residual': 0,
        'has_dense': 0,
        'has_inception': 0,
    },
    'densenet121': {
        'num_conv_layers': 121,
        'total_params': 7_978_856,
        'total_conv_params': 7_036_416,
        'depth': 121,
        'max_channels': 1024,
        'has_residual': 0,
        'has_dense': 1,
        'has_inception': 0,
    },
    'resnet152': {
        'num_conv_layers': 155,
        'total_params': 60_192_808,
        'total_conv_params': 58_143_808,
        'depth': 152,
        'max_channels': 2048,
        'has_residual': 1,
        'has_dense': 0,
        'has_inception': 0,
    },
    'googlenet': {
        'num_conv_layers': 57,
        'total_params': 6_624_904,
        'total_conv_params': 6_166_016,
        'depth': 22,
        'max_channels': 1024,
        'has_residual': 0,
        'has_dense': 0,
        'has_inception': 1,
    },
    'alexnet': {
        'num_conv_layers': 5,
        'total_params': 61_100_840,
        'total_conv_params': 3_747_200,
        'depth': 8,
        'max_channels': 256,
        'has_residual': 0,
        'has_dense': 0,
        'has_inception': 0,
    },
    'resnet50': {
        'num_conv_layers': 53,
        'total_params': 25_557_032,
        'total_conv_params': 23_508_032,
        'depth': 50,
        'max_channels': 2048,
        'has_residual': 1,
        'has_dense': 0,
        'has_inception': 0,
    },
}

MODEL_ARCH_FEATURE_NAMES = list(
    next(iter(_MODEL_ARCH_FEATURES.values())).keys()
)


def get_model_arch_features(model_name: str) -> Dict[str, float]:
    key = model_name.lower()
    if key in _MODEL_ARCH_FEATURES:
        return _MODEL_ARCH_FEATURES[key]
    raise KeyError(
        f"Unknown model '{model_name}'. "
        f"Available: {list(_MODEL_ARCH_FEATURES.keys())}"
    )


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add architecture-level features to a model-level (overall) DataFrame.

    Expected input columns: model, algorithm, input_size, device, ...

    Added columns: all MODEL_ARCH_FEATURE_NAMES + all DEVICE_FEATURE_NAMES
    """
    df = df.copy()

    for feat in MODEL_ARCH_FEATURE_NAMES:
        df[feat] = df['model'].map(
            lambda m, f=feat: _MODEL_ARCH_FEATURES.get(m, {}).get(f, 0)
        )

    df = _add_device_features(df)

    return df


# ============================================================================
# SHARED HELPERS
# ============================================================================

def _add_device_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the 'device' column into numerical GPU spec columns."""
    for feat in DEVICE_FEATURE_NAMES:
        df[feat] = df['device'].map(
            lambda d, f=feat: DEVICE_SPECS.get(d, {}).get(f, 0)
        )
    return df


def load_layer_data(
    csv_path: str,
    drop_model_col: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load layer CSV + engineer features.

    Returns (X, y) where y = log1p(mean_ms).
    """
    df = pd.read_csv(csv_path)

    if drop_model_col and 'model' in df.columns:
        df = df.drop(columns=['model'])

    if 'layer' in df.columns:
        df = df.drop(columns=['layer'])

    target_col = 'mean_ms'
    y = np.log1p(df.pop(target_col))

    extra_targets = ['std_ms', 'percentage']
    for col in extra_targets:
        if col in df.columns:
            df = df.drop(columns=[col])

    energy_cols = ['power_mean_w', 'power_std_w',
                   'energy_per_layer_j', 'power_samples',
                   'energy_attribution']
    energy_data = {}
    for col in energy_cols:
        if col in df.columns:
            energy_data[col] = df.pop(col)

    df = add_layer_features(df)

    return df, y, energy_data


def load_overall_data(
    csv_path: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load overall CSV + engineer features.

    Returns (X, y_time, y_energy) where
        y_time   = log1p(mean)                  -- overall latency ms
        y_energy = log1p(energy_per_inference_j) -- overall energy J
    """
    df = pd.read_csv(csv_path)

    y_time = np.log1p(df.pop('mean'))

    y_energy = None
    if 'energy_per_inference_j' in df.columns:
        y_energy = np.log1p(df.pop('energy_per_inference_j'))

    drop_cols = [
        'std', 'min', 'max', 'median',
        'power_mean_w', 'power_std_w', 'power_min_w', 'power_max_w',
        'energy_total_j', 'power_samples', 'batch_size',
    ]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = add_model_features(df)

    return df, y_time, y_energy
