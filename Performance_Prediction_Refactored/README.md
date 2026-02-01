# Performance Prediction - Refactored

A unified GPU performance profiling system for the ai3 library. This refactored version consolidates all model/algorithm combinations into a single, reusable codebase.

## Overview

This replaces the individual per-model/per-algorithm scripts with:
- **One profiler module** (`profiler.py`) containing all core classes
- **One CLI entry point** (`run_profile.py`) that accepts model/algorithm as arguments
- **Three staged job scripts** for progressive testing

## Quick Start

### 1. Sanity Check (5 minutes)
Verify the combination works before committing GPU hours:
```bash
./1_sanity_check.sh vgg16 gemm l40s
```

### 2. Quick Sweep (30-60 minutes)
Test 10 input sizes to verify scaling looks correct:
```bash
./2_quick_sweep.sh vgg16 gemm l40s
```

### 3. Full Run (2-6 hours)
Collect complete data with 100 input sizes:
```bash
./3_full_run.sh vgg16 gemm l40s
```

### Check Progress
```bash
./check_status.sh
```

## Available Options

### Models
- `vgg16`
- `densenet121`
- `resnet152`
- `googlenet`
- `alexnet`
- `resnet50`

### Algorithms
- `direct` - Straightforward convolution
- `gemm` - General Matrix Multiplication (im2col + matmul)
- `implicit gemm` - GEMM without explicit im2col
- `implicit precomp gemm` - Implicit GEMM with precomputed indices
- `fft` - Fast Fourier Transform
- `fft tiling` - FFT with tiled computation
- `winograd` - Winograd transform (VGG16 only, 3x3 kernels)
- `winograd nonfused` - Winograd with separate stages (VGG16 only)
- `guess` - cuDNN auto-selection

### GPUs
- `l40s` - NVIDIA L40S (recommended)
- `h100` - NVIDIA H100
- `v100` - NVIDIA V100

## Direct Python Usage

You can also run the profiler directly:

```bash
# Test mode
python run_profile.py --model vgg16 --algorithm gemm --test

# Quick mode  
python run_profile.py --model vgg16 --algorithm gemm --quick

# Full mode
python run_profile.py --model vgg16 --algorithm gemm --num-sizes 100

# Custom settings
python run_profile.py --model densenet121 --algorithm direct \
    --num-sizes 50 --warmup 5 --measure 15 --output-dir ./my_results

# List options
python run_profile.py --list-models
python run_profile.py --list-algorithms
python run_profile.py --check-cuda
```

## Directory Structure

```
Performance_Prediction_Refactored/
├── profiler.py              # Core profiling classes
│   ├── PowerMonitor         # GPU power monitoring (NVML)
│   ├── CUDALayerTimer       # Layer-wise timing with hooks
│   └── UnifiedProfiler      # Main profiler orchestrator
│
├── run_profile.py           # CLI entry point
│
├── 1_sanity_check.sh        # Stage 1: Quick verification
├── 2_quick_sweep.sh         # Stage 2: 10 input sizes
├── 3_full_run.sh            # Stage 3: Full 100 sizes
├── check_status.sh          # View progress
│
├── results/                 # All outputs organized here
│   ├── 1_sanity/
│   │   └── vgg16_gemm_l40s/
│   │       ├── vgg16_gemm_overall.csv
│   │       └── vgg16_gemm_layers.csv
│   ├── 2_quick/
│   └── 3_full/
│
└── README.md
```

## Output Files

Each run produces two CSV files:

### `*_overall.csv`
Overall model performance per input size:
| Column | Description |
|--------|-------------|
| model | Model name |
| algorithm | Algorithm used |
| input_size | Input dimension (e.g., 224, 256, ...) |
| mean | Mean inference time (ms) |
| std | Standard deviation (ms) |
| min/max | Min and max times |
| power_mean_w | Average power (Watts) |
| energy_per_inference_j | Energy per inference (Joules) |

### `*_layers.csv`
Per-layer breakdown:
| Column | Description |
|--------|-------------|
| layer | Layer name |
| in_channels | Input channels |
| out_channels | Output channels |
| kernel_size | Kernel dimensions |
| mean_ms | Mean layer time (ms) |
| percentage | Percentage of total time |

## Features

### Accurate GPU Timing
Uses CUDA events for precise GPU timing:
```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
# ... inference ...
end_event.record()
torch.cuda.synchronize()
duration = start_event.elapsed_time(end_event)
```

### Power Monitoring
Uses NVML for real power measurements with trapezoidal integration:
```python
# Continuous sampling during inference
# Energy = ∫P(t)dt ≈ Σ[(P_i + P_{i+1})/2] × Δt
```

### Layer-wise Profiling
PyTorch hooks capture timing for each convolution layer individually, allowing you to identify bottlenecks.

## Workflow Example

```bash
# Step 1: Test that VGG16 + GEMM works
./1_sanity_check.sh vgg16 gemm l40s
# Wait for job...
cat results/1_sanity/vgg16_gemm_l40s/*.out
# ✓ Looks good!

# Step 2: Verify scaling behavior
./2_quick_sweep.sh vgg16 gemm l40s
# Wait for job...
head results/2_quick/vgg16_gemm_l40s/*_overall.csv
# ✓ Times increase with input size as expected!

# Step 3: Collect full data
./3_full_run.sh vgg16 gemm l40s
# Go get coffee ☕

# Repeat for other combinations...
./1_sanity_check.sh vgg16 winograd l40s
./1_sanity_check.sh densenet121 gemm l40s
# ...
```

## Troubleshooting

### "cuDNN is not available"
Rebuild ai3 with the `USE_CUDNN` flag enabled.

### "Algorithm X not compatible with model Y"
Winograd algorithms only work with VGG16 (requires 3x3 kernels throughout).

### Job stuck in queue
Check available GPUs: `sinfo -C gpul40s`

### Out of memory
Reduce batch size or use a GPU with more memory (H100 has 80GB).

## Comparison with Original Structure

| Original | Refactored |
|----------|------------|
| ~50 separate Python files | 1 profiler module |
| ~50 separate job scripts | 3 staged scripts |
| Copy-paste for new combos | Just pass arguments |
| Scattered results | Organized `results/` folder |
| Edit each file for changes | Edit once |

## Requirements

- Python 3.9+
- PyTorch 2.4+
- torchvision
- ai3 (with cuDNN support for most algorithms)
- nvidia-ml-py3 (for power monitoring)

Install power monitoring:
```bash
pip install nvidia-ml-py3
```
