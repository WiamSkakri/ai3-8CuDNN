# HPC GPU Resources Guide

This document provides information about available GPU resources on the HPC cluster and how to request them in your job scripts.

## Available GPU Resources

### GPU Types Overview

| GPU Type | Feature Flag | Nodes | GPUs per Node | Total GPUs | Best For |
|----------|--------------|-------|---------------|------------|----------|
| **NVIDIA L40S** | `gpul40s` | `gput[063-071]` | 4 | 36 | **Deep learning, cuDNN workloads** |
| **NVIDIA H100** | `gpu2h100` | `gput[072-073]` | 2 | 4 | High-performance computing, large models |
| **NVIDIA RTX 4090** | `gpu4090` | `gput[074-075]` | 4 | 8 | Gaming architecture, good for research |
| **NVIDIA V100 (4x)** | `gpu4v100` | `gput[053-056]` | 4 | 16 | Legacy deep learning workloads |
| **NVIDIA V100 (2x)** | `gpu2v100` | `gput[057-062]` | 2 | 12 | Legacy deep learning workloads |
| **NVIDIA RTX 2080** | `gpu2080` | `gput[045-051]` | 2 | 14 | General purpose GPU computing |
| **NVIDIA P100** | `gpup100` | `gput[034-044]` | 2 | 22 | Older generation, basic GPU tasks |
| **DGX System** | `dgx` | `dgxt001` | 8 | 8 | Special DGX node |
| **AI Cluster** | N/A | `aisct[01-04]` | 8 | 32 | AI-specific nodes |
| **AI Cluster IT** | N/A | `aisciit[01-05]` | 8 | 40 | AI-specific IT nodes |

### Detailed GPU Specifications

#### NVIDIA L40S (Recommended for cuDNN Performance Prediction)
- **Architecture**: Ada Lovelace
- **CUDA Cores**: 18,176
- **Tensor Cores**: 568 (4th gen)
- **Memory**: 48 GB GDDR6
- **Memory Bandwidth**: 864 GB/s
- **FP32 Performance**: 91.6 TFLOPS
- **Tensor Performance**: 733 TFLOPS (FP16)
- **Best Use**: Deep learning training/inference, cuDNN algorithm profiling

#### NVIDIA H100
- **Architecture**: Hopper
- **Memory**: 80 GB HBM3
- **Tensor Performance**: 2000 TFLOPS (FP16)
- **Best Use**: Very large models, highest performance workloads

#### NVIDIA RTX 4090
- **Architecture**: Ada Lovelace
- **CUDA Cores**: 16,384
- **Memory**: 24 GB GDDR6X
- **Best Use**: Research, smaller models

#### NVIDIA V100
- **Architecture**: Volta
- **Memory**: 16 or 32 GB HBM2
- **Best Use**: Legacy workloads, good cuDNN support

#### NVIDIA RTX 2080
- **Architecture**: Turing
- **CUDA Cores**: 2,944
- **Memory**: 8 GB GDDR6
- **Best Use**: Basic GPU computing, development

#### NVIDIA P100
- **Architecture**: Pascal
- **Memory**: 16 GB HBM2
- **Best Use**: Older generation tasks

## How to Request GPUs in SLURM

### Method 1: Using Feature Flags (Recommended)

Request a specific GPU type using the `-C` (constraint) flag:

```bash
#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH -C gpul40s                # Request L40S specifically
#SBATCH --time=01:00:00
#SBATCH --mem=32G
```

### Method 2: Using Node Lists

Request from specific nodes:

```bash
#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --nodelist=gput063        # Specific node
#SBATCH --gres=gpu:2              # Request 2 GPUs
#SBATCH --time=01:00:00
```

### Method 3: Exclude Unwanted GPUs

Avoid certain GPU types:

```bash
#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --gres=gpu:1
#SBATCH --exclude=gput[034-044]   # Exclude P100 nodes
#SBATCH --time=01:00:00
```

## Feature Flag Reference

Use these flags with `#SBATCH -C <flag>`:

| Flag | GPU Type | Recommended For |
|------|----------|-----------------|
| `gpul40s` | NVIDIA L40S | **cuDNN performance testing, VGG16 profiling** |
| `gpu2h100` | NVIDIA H100 | Largest models, maximum performance |
| `gpu4090` | NVIDIA RTX 4090 | Research, medium-scale training |
| `gpu4v100` | NVIDIA V100 (4 GPUs) | Legacy deep learning |
| `gpu2v100` | NVIDIA V100 (2 GPUs) | Legacy deep learning |
| `gpu2080` | NVIDIA RTX 2080 | Development, testing |
| `gpup100` | NVIDIA P100 | Basic tasks |
| `dgx` | DGX System | Special access required |

## Common SLURM Commands

### Check Available GPUs
```bash
# See all GPU nodes and availability
sinfo -o "%20N %10c %10m %25f %10G"

# Check specific GPU type availability
sinfo -C gpul40s

# See current GPU node status
sinfo -p gpu --long
```

### Check Your Current Job
```bash
# See your job's GPU allocation
scontrol show job $SLURM_JOB_ID | grep GRES

# Check which node you're on
echo $SLURMD_NODENAME
```

### Check Specific Node Details
```bash
# See details of a specific node
scontrol show node gput063

# See GPU allocation on a node
scontrol show node gput063 | grep Gres
```

### Interactive GPU Session
```bash
# Request interactive session with L40S GPU
srun -C gpul40s --gres=gpu:1 --time=02:00:00 --mem=32G --pty bash

# Request interactive session with specific resources
srun --nodelist=gput063 --gres=gpu:1 --cpus-per-task=8 --time=02:00:00 --pty bash
```

## GPU Monitoring Commands

Once you're on a GPU node:

```bash
# Basic GPU information
nvidia-smi

# Continuous monitoring (updates every 1 second)
watch -n 1 nvidia-smi

# Detailed GPU query
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv

# List all GPUs
nvidia-smi -L

# Check CUDA version
nvcc --version
```

## Recommendations for VGG16 Performance Prediction

For your cuDNN algorithm profiling work (GEMM, Implicit GEMM, Winograd, etc.):

### Primary Choice: **NVIDIA L40S** (`gpul40s`)
```bash
#SBATCH -C gpul40s
#SBATCH --gres=gpu:1
```
- Latest Ada Lovelace architecture
- Excellent cuDNN 8.x/9.x support
- Good for profiling modern algorithms
- 48 GB memory (handles large batch sizes)

### Alternative: **NVIDIA H100** (`gpu2h100`)
```bash
#SBATCH -C gpu2h100
#SBATCH --gres=gpu:1
```
- Latest Hopper architecture
- Best performance for newer cuDNN versions
- Limited availability (only 4 GPUs)

### Development/Testing: **NVIDIA RTX 2080** (`gpu2080`)
```bash
#SBATCH -C gpu2080
#SBATCH --gres=gpu:1
```
- More available
- Good for quick testing
- Lower performance but sufficient for algorithm selection studies

## Example Job Scripts

### Example 1: Single L40S GPU Job
```bash
#!/bin/bash
#SBATCH --job-name=vgg16_gemm
#SBATCH --output=logs/gemm_%j.out
#SBATCH --error=logs/gemm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -C gpul40s
#SBATCH --time=04:00:00
#SBATCH --mem=64G

# Load modules
module load cuda/11.8

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ai3_8cudnn

# Run profiling
cd $SLURM_SUBMIT_DIR
python vgg16_gemm.py
```

### Example 2: Multi-GPU Job
```bash
#!/bin/bash
#SBATCH --job-name=multi_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH -C gpul40s
#SBATCH --time=08:00:00
#SBATCH --mem=128G

# Run with multiple GPUs
srun python multi_gpu_script.py
```

### Example 3: Array Job Across Different GPUs
```bash
#!/bin/bash
#SBATCH --job-name=gpu_sweep
#SBATCH --array=1-5
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

case $SLURM_ARRAY_TASK_ID in
    1) GPU_TYPE="gpul40s" ;;
    2) GPU_TYPE="gpu2h100" ;;
    3) GPU_TYPE="gpu4090" ;;
    4) GPU_TYPE="gpu2v100" ;;
    5) GPU_TYPE="gpu2080" ;;
esac

#SBATCH -C $GPU_TYPE

python benchmark_gpu.py --gpu-type=$GPU_TYPE
```

## Troubleshooting

### Job Pending with Reason "Resources"
```bash
# Check why your job is waiting
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %.20R"

# Solution: Try a different GPU type or reduce resource requirements
```

### No GPUs Visible in Job
```bash
# Make sure you requested GPUs
#SBATCH --gres=gpu:1

# Check if CUDA_VISIBLE_DEVICES is set
echo $CUDA_VISIBLE_DEVICES
```

### Out of Memory Errors
```bash
# Request more system memory
#SBATCH --mem=64G

# Or use GPUs with more memory (L40S: 48GB, H100: 80GB)
#SBATCH -C gpul40s
```

## Project-Specific Notes

For the VGG16 performance prediction project in this repository:

1. **GEMM Algorithm**: Use L40S for best matmul performance
2. **Implicit GEMM**: Use L40S or H100 for tensor core utilization
3. **Winograd**: Use L40S (good for transform-based convolutions)
4. **Algorithm Comparison**: Run on same GPU type (L40S recommended) for fair comparison

## Additional Resources

- [SLURM Documentation](https://slurm.schedmd.com/)
- [NVIDIA GPU Specs](https://www.nvidia.com/en-us/data-center/)
- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)

---

**Last Updated**: October 27, 2024  
**Cluster**: HPC8  
**Contact**: Check with HPC admin for access to restricted nodes (DGX, AI clusters)

