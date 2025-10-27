# GoogLeNet Implicit GEMM Profiling

Complete performance profiling for GoogLeNet using ai3's Implicit GEMM algorithm.

## Files

- `googlenet_implicit_gemm.py` - Main profiling script (100 input sizes, 224-512 pixels)
- `verify_implicit_gemm.py` - Verification script to confirm ai3 Implicit GEMM is being used
- `run_googlenet_implicit_gemm.sh` - SLURM batch script for HPC submission (V100 GPU)
- `job_l40.sh` - SLURM batch script for HPC submission (L40 GPU)

## Quick Start on HPC

### 1. Navigate to this directory
```bash
cd Performance_Prediction/googlenet/implicit_gemm
```

### 2. Submit the job
```bash
sbatch run_googlenet_implicit_gemm.sh
```

For L40 GPUs:
```bash
sbatch job_l40.sh
```

### 3. Check job status
```bash
squeue -u $USER
```

## Configuration

Default settings in `googlenet_implicit_gemm.py`:

```python
BATCH_SIZE = 1           # Batch size
WARMUP_ITERS = 10       # Warmup iterations (discarded)
MEASURE_ITERS = 20      # Measurement iterations (used for stats)
INPUT_SIZES = 100       # Number of input sizes to test
```
