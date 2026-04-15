# Layers -- Time and Energy Prediction System

Predicts **latency (time)** and **energy** at both the **layer level** and **model level** for CNN inference with different cuDNN convolution algorithms across GPUs.

## Overview

| Task | Target | Level | Data Source |
|------|--------|-------|-------------|
| Layer Time | `mean_ms` | Per conv layer | `combined_layers.csv` |
| Layer Energy | `energy_per_layer_j` | Per conv layer | `combined_layers.csv` (v2 profiler) |
| Model Time | `mean` (overall ms) | Whole model | `combined_overall.csv` |
| Model Energy | `energy_per_inference_j` | Whole model | `combined_overall.csv` |

Each task trains 4 ML models: **XGBoost**, **LightGBM**, **Random Forest**, **Neural Network (MLP)**.

## Directory Structure

```
Layers/
  profiler.py              # v2 profiler with per-layer NVML power sampling
  run_profile.py           # CLI entry point for profiling
  device_specs.py          # GPU hardware specs lookup table
  feature_engineering.py   # Shared feature engineering (layer + model level)
  train_layer_time.py      # Layer-level latency prediction
  train_layer_energy.py    # Layer-level energy prediction
  train_model_time.py      # Model-level latency prediction
  train_model_energy.py    # Model-level energy prediction
  evaluate.py              # Unified evaluation across all 4 tasks
  run_profile_full.sh      # SLURM: profile one model/algo/gpu combo
  run_profile_all.sh       # SLURM: submit all profiling combos
  run_train_all.sh         # SLURM: train all predictors
  data/                    # Place combined CSVs here
  artifacts/               # Saved .joblib model artifacts
```

## Quick Start

### 1. Prepare data

Copy or symlink the existing profiling data:

```bash
cp ../Performance_Prediction_Refactored/results/combined_layers.csv data/
cp ../Performance_Prediction_Refactored/results/combined_overall.csv data/
```

### 2. Train layer-level time predictor (works with existing data)

```bash
python train_layer_time.py
```

### 3. Train model-level time + energy predictors (works with existing data)

```bash
python train_model_time.py
python train_model_energy.py
```

### 4. Re-profile with v2 profiler for per-layer energy

```bash
# Single combo
./run_profile_full.sh vgg16 gemm l40s

# All combos
./run_profile_all.sh
```

### 5. Train layer-level energy predictor (requires v2 profiler data)

```bash
python train_layer_energy.py
```

### 6. Run unified evaluation

```bash
python evaluate.py
```

## Features

### Profiler v2 (per-layer power/energy)
- Continuous NVML background sampling during layer measurement
- Wall-clock timestamp markers in forward hooks
- Post-hoc slicing of power samples per layer
- Trapezoidal rule energy integration per layer
- New CSV columns: `power_mean_w`, `power_std_w`, `energy_per_layer_j`, `power_samples`, `energy_attribution` (`nvml_trapezoid` | `nvml_single_sample` | `proportional_overall`)

### Feature Engineering
- **Layer-level**: `compute_flops`, `output_size`, `params`, `memory_bytes`, `compute_intensity`
- **Model-level**: `num_conv_layers`, `total_params`, `depth`, `max_channels`, `has_residual`, `has_dense`, `has_inception`
- **Device**: Numerical GPU specs (bandwidth, TFLOPS, memory, TDP) instead of one-hot encoding

### Generalization
- Architecture-level features enable prediction on unseen models
- Device hardware specs enable prediction on unseen GPUs
- Leave-one-model-out cross-validation tests generalization

## Requirements

- Python 3.9+
- PyTorch 2.4+, torchvision
- ai3 (with cuDNN support)
- scikit-learn, xgboost, lightgbm, numpy, pandas, joblib
- nvidia-ml-py3 (for power monitoring during profiling)
