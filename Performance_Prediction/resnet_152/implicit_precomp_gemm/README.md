# ResNet152 Implicit Precomp GEMM Profiling

Complete performance profiling for ResNet152 using ai3's Implicit Precomp GEMM algorithm.

## Files

- `resnet_152_implicit_precomp_gemm.py` - Main profiling script (100 input sizes, 224-512 pixels)
- `verify_implicit_precomp_gemm.py` - Verification script to confirm ai3 Implicit Precomp GEMM is being used
- `run_resnet_152_implicit_precomp_gemm.sh` - SLURM batch script for HPC submission (V100 GPU)
- `job_l40.sh` - SLURM batch script for HPC submission (L40 GPU)

## Quick Start on HPC

### 1. Navigate to this directory
```bash
cd Performance_Prediction/resnet_152/implicit_precomp_gemm
```

### 2. Submit the job
For V100 GPUs:
```bash
sbatch run_resnet_152_implicit_precomp_gemm.sh
```

For L40 GPUs:
```bash
sbatch job_l40.sh
```

### 3. Check job status
```bash
squeue -u $USER
```

### 4. Monitor progress (once running)
```bash
tail -f resnet_152_implicit_precomp_gemm_results/resnet_152_implicit_precomp_gemm_<JOBID>.out
```

## What the Script Does

The batch script:
- ✅ Requests **1 GPU** (V100 or L40 depending on script)
- ✅ Allocates 16GB RAM, 4 CPUs
- ✅ Sets 24-hour time limit
- ✅ Activates `ai3_8cudnn` conda environment
- ✅ Runs in the directory where you submit it (`$SLURM_SUBMIT_DIR`)
- ✅ Looks for `resnet_152_implicit_precomp_gemm.py` in the same directory

## Expected Runtime

- **~25-35 minutes** for full profiling (100 input sizes)
- **~30 seconds** for verification test

## Output Files

After completion, all results will be in the `resnet_152_implicit_precomp_gemm_results/` directory:

```
resnet_152_implicit_precomp_gemm_results/
├── resnet_152_implicit_precomp_gemm_<JOBID>.out          # Standard output log
├── resnet_152_implicit_precomp_gemm_<JOBID>.err          # Error log (should be empty)
├── ResNet152_implicit_precomp_gemm_cuda_overall.csv     # Overall performance (100 rows)
└── ResNet152_implicit_precomp_gemm_cuda_layers.csv      # Layer-wise performance (~1300 rows)
```

The directory is automatically created by the bash script.

## GPU Specification

The `run_resnet_152_implicit_precomp_gemm.sh` script requests:
```bash
#SBATCH --partition=gpu
#SBATCH -C gpu2v100           # V100 GPU constraint
#SBATCH --gres=gpu:1          # 1 GPU
```

The `job_l40.sh` script requests:
```bash
#SBATCH --partition=gpu
#SBATCH -C gpul40s            # L40 GPU constraint
#SBATCH --gres=gpu:1          # 1 GPU
```

If your HPC has different GPU naming, you may need to adjust:
- Check available GPUs: `sinfo -o "%20P %10g %10G"`
- Check available partitions: `sinfo`

## Verify Before Full Run

To test with just 1 input size (30 seconds):

1. Edit `resnet_152_implicit_precomp_gemm.py` line 303:
```python
INPUT_SIZES = [224 + int(i * (512 - 224) / 99) for i in range(1)]  # Test
```

2. Run interactively:
```bash
srun --gres=gpu:1 --partition=gpu --pty bash
conda activate ai3_8cudnn
python resnet_152_implicit_precomp_gemm.py
```

3. Change back to 100 for full run:
```python
INPUT_SIZES = [224 + int(i * (512 - 224) / 99) for i in range(100)]  # Full
```

## Verify ai3 Implicit Precomp GEMM is Being Used

Run the verification script to confirm:

```bash
python verify_implicit_precomp_gemm.py
```

Expected output:
```
✓ All 50+ Conv2D layers were replaced with ai3 Implicit Precomp GEMM implementation
✓ Layer inspection confirms ai3 algorithm='implicit precomp gemm' is active
✓ Performance difference confirms different execution
→ CONFIRMED: Your profiling is using ai3 Implicit Precomp GEMM, NOT PyTorch default!
```

## Troubleshooting

### Job fails immediately
Check error log:
```bash
cat resnet_152_implicit_precomp_gemm_results/resnet_152_implicit_precomp_gemm_<JOBID>.err
```

### "resnet_152_implicit_precomp_gemm.py not found"
Make sure you submit from the correct directory:
```bash
cd Performance_Prediction/resnet_152/implicit_precomp_gemm
sbatch run_resnet_152_implicit_precomp_gemm.sh
```

### GPU not available
Check if gpu partition exists:
```bash
sinfo | grep gpu
```

If not, adjust partition name in the script.

### Environment activation fails
Check your conda setup:
```bash
conda env list | grep ai3
```

If environment name is different, update line 38 in the script.

## Results Analysis

Load and analyze results:

```python
import pandas as pd

# Load results from the results directory
overall = pd.read_csv('resnet_152_implicit_precomp_gemm_results/ResNet152_implicit_precomp_gemm_cuda_overall.csv')
layers = pd.read_csv('resnet_152_implicit_precomp_gemm_results/ResNet152_implicit_precomp_gemm_cuda_layers.csv')

# Plot performance vs input size
import matplotlib.pyplot as plt
plt.plot(overall['input_size'], overall['mean_ms'])
plt.xlabel('Input Size (pixels)')
plt.ylabel('Execution Time (ms)')
plt.title('ResNet152 Implicit Precomp GEMM Performance')
plt.show()

# Find slowest layers
slowest = layers.groupby('layer')['mean_ms'].mean().sort_values(ascending=False).head(10)
print(slowest)
```

## Configuration

Default settings in `resnet_152_implicit_precomp_gemm.py`:

```python
BATCH_SIZE = 1           # Batch size
WARMUP_ITERS = 10       # Warmup iterations (discarded)
MEASURE_ITERS = 20      # Measurement iterations (used for stats)
INPUT_SIZES = 100       # Number of input sizes to test
```

Adjust these if needed for faster/more thorough profiling.

## About Implicit Precomp GEMM

Implicit Precomp GEMM is a cuDNN convolution algorithm that:
- Performs direct convolution using implicit matrix multiplication
- Precomputes certain transformations for optimization
- May offer performance advantages on specific GPU architectures
- Particularly optimized for modern GPUs with Tensor Core support

This algorithm is part of cuDNN's suite of convolution implementations and is automatically selected by cuDNN based on heuristics, but ai3 allows you to explicitly choose it for benchmarking and comparison.

