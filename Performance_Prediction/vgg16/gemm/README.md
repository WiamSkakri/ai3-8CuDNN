# VGG16 GEMM Profiling

Complete performance profiling for VGG16 using ai3's GEMM algorithm.

## Files

- `vgg16_gemm.py` - Main profiling script (100 input sizes, 224-512 pixels)
- `verify_gemm.py` - Verification script to confirm ai3 GEMM is being used
- `run_vgg16_gemm.sh` - SLURM batch script for HPC submission

## Quick Start on HPC

### 1. Navigate to this directory
```bash
cd Performance_Prediction/vgg16/gemm
```

### 2. Submit the job
```bash
sbatch run_vgg16_gemm.sh
```

### 3. Check job status
```bash
squeue -u $USER
```

### 4. Monitor progress (once running)
```bash
tail -f vgg16_gemm_results/vgg16_gemm_<JOBID>.out
```

## What the Script Does

The batch script (`run_vgg16_gemm.sh`):
- ✅ Requests **1 V100 GPU** on **gpu2v100** partition
- ✅ Allocates 16GB RAM, 4 CPUs
- ✅ Sets 24-hour time limit
- ✅ Activates `ai3_8cudnn` conda environment
- ✅ Runs in the directory where you submit it (`$SLURM_SUBMIT_DIR`)
- ✅ Looks for `vgg16_gemm.py` in the same directory

## Expected Runtime

- **~25-35 minutes** for full profiling (100 input sizes)
- **~30 seconds** for verification test

## Output Files

After completion, all results will be in the `vgg16_gemm_results/` directory:

```
vgg16_gemm_results/
├── vgg16_gemm_<JOBID>.out          # Standard output log
├── vgg16_gemm_<JOBID>.err          # Error log (should be empty)
├── VGG16_gemm_cuda_overall.csv     # Overall performance (100 rows)
└── VGG16_gemm_cuda_layers.csv      # Layer-wise performance (~1300 rows)
```

The directory is automatically created by the bash script.

## GPU Specification

The script requests:
```bash
#SBATCH --gres=gpu:v100:1       # 1 V100 GPU
#SBATCH --partition=gpu2v100    # gpu2v100 partition
```

If your HPC has different GPU naming, you may need to adjust:
- Check available GPUs: `sinfo -o "%20P %10g %10G"`
- Check available partitions: `sinfo`

## Verify Before Full Run

To test with just 1 input size (30 seconds):

1. Edit `vgg16_gemm.py` line 303:
```python
INPUT_SIZES = [224 + int(i * (512 - 224) / 99) for i in range(1)]  # Test
```

2. Run interactively:
```bash
srun --gres=gpu:v100:1 --partition=gpu2v100 --pty bash
conda activate ai3_8cudnn
python vgg16_gemm.py
```

3. Change back to 100 for full run:
```python
INPUT_SIZES = [224 + int(i * (512 - 224) / 99) for i in range(100)]  # Full
```

## Verify ai3 GEMM is Being Used

Run the verification script to confirm:

```bash
python verify_gemm.py
```

Expected output:
```
✓ All 13 Conv2D layers were replaced with ai3 GEMM implementation
✓ Layer inspection confirms ai3 algorithm='gemm' is active
✓ Performance difference (-35%) confirms different execution
→ CONFIRMED: Your profiling is using ai3 GEMM, NOT PyTorch default!
```

## Troubleshooting

### Job fails immediately
Check error log:
```bash
cat vgg16_gemm_<JOBID>.err
```

### "vgg16_gemm.py not found"
Make sure you submit from the correct directory:
```bash
cd Performance_Prediction/vgg16/gemm
sbatch run_vgg16_gemm.sh
```

### GPU not available
Check if gpu2v100 partition exists:
```bash
sinfo | grep gpu2v100
```

If not, adjust partition name in `run_vgg16_gemm.sh`.

### Environment activation fails
Check your conda setup:
```bash
conda env list | grep ai3
```

If environment name is different, update line 38 in `run_vgg16_gemm.sh`.

## Results Analysis

Load and analyze results:

```python
import pandas as pd

# Load results from the results directory
overall = pd.read_csv('vgg16_gemm_results/VGG16_gemm_cuda_overall.csv')
layers = pd.read_csv('vgg16_gemm_results/VGG16_gemm_cuda_layers.csv')

# Plot performance vs input size
import matplotlib.pyplot as plt
plt.plot(overall['input_size'], overall['mean_ms'])
plt.xlabel('Input Size (pixels)')
plt.ylabel('Execution Time (ms)')
plt.title('VGG16 GEMM Performance')
plt.show()

# Find slowest layers
slowest = layers.groupby('layer')['mean_ms'].mean().sort_values(ascending=False).head(10)
print(slowest)
```

## Configuration

Default settings in `vgg16_gemm.py`:

```python
BATCH_SIZE = 1           # Batch size
WARMUP_ITERS = 10       # Warmup iterations (discarded)
MEASURE_ITERS = 20      # Measurement iterations (used for stats)
INPUT_SIZES = 100       # Number of input sizes to test
```

Adjust these if needed for faster/more thorough profiling.

