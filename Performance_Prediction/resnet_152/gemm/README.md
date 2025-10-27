# ResNet152 GEMM Profiling

Complete performance profiling for ResNet152 using ai3's GEMM algorithm.

## Files

- `resnet_152_gemm.py` - Main profiling script (100 input sizes, 224-512 pixels)
- `verify_gemm.py` - Verification script to confirm ai3 GEMM is being used
- `run_resnet_152_gemm.sh` - SLURM batch script for HPC submission (V100 GPU)
- `job_l40.sh` - SLURM batch script for HPC submission (L40 GPU)
- `h100_job.sh` - SLURM batch script for HPC submission (H100 GPU)

## Quick Start on HPC

### 1. Navigate to this directory
```bash
cd Performance_Prediction/resnet_152/gemm
```

### 2. Submit the job
For V100 GPUs:
```bash
sbatch run_resnet_152_gemm.sh
```

For L40 GPUs:
```bash
sbatch job_l40.sh
```

For H100 GPUs:
```bash
sbatch h100_job.sh
```

### 3. Check job status
```bash
squeue -u $USER
```

### 4. Monitor progress (once running)
```bash
tail -f resnet152_gemm_results_*/resnet152_gemm_*_<JOBID>.out
```

## What the Script Does

The batch script:
- ✅ Requests **1 GPU** (V100, L40, or H100 depending on script)
- ✅ Allocates 16GB RAM, 4 CPUs
- ✅ Sets 24-hour time limit
- ✅ Activates `ai3_8cudnn` conda environment
- ✅ Runs in the directory where you submit it (`$SLURM_SUBMIT_DIR`)
- ✅ Looks for `resnet_152_gemm.py` in the same directory

## Expected Runtime

- **~30-40 minutes** for full profiling (100 input sizes)
- **~30 seconds** for verification test

## Output Files

After completion, all results will be in the `resnet152_gemm_results_*/` directory:

```
resnet152_gemm_results_*/
├── resnet152_gemm_*_<JOBID>.out          # Standard output log
├── resnet152_gemm_*_<JOBID>.err          # Error log (should be empty)
├── ResNet152_gemm_cuda_overall.csv       # Overall performance (100 rows)
└── ResNet152_gemm_cuda_layers.csv        # Layer-wise performance
```

The directory is automatically created by the bash script.

## Verify ai3 GEMM is Being Used

Run the verification script to confirm:

```bash
python verify_gemm.py
```

Expected output:
```
✓ All Conv2D layers were replaced with ai3 GEMM implementation
✓ Layer inspection confirms ai3 algorithm='gemm' is active
→ CONFIRMED: Your profiling is using ai3 GEMM, NOT PyTorch default!
```

## Configuration

Default settings in `resnet_152_gemm.py`:

```python
BATCH_SIZE = 1           # Batch size
WARMUP_ITERS = 10       # Warmup iterations (discarded)
MEASURE_ITERS = 20      # Measurement iterations (used for stats)
INPUT_SIZES = 100       # Number of input sizes to test
```

Adjust these if needed for faster/more thorough profiling.


