#!/bin/bash
#SBATCH --job-name=resnet_152_fft_tiling_l40s
#SBATCH --output=resnet_152_fft_tiling_results_l40s/resnet_152_fft_tiling_l40s_%j.out
#SBATCH --error=resnet_152_fft_tiling_results_l40s/resnet_152_fft_tiling_l40s_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH -C gpul40s
#SBATCH --gres=gpu:1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "=========================================="
nvidia-smi

source ~/.bashrc
conda activate ai3_8cudnn

cd $SLURM_SUBMIT_DIR
[ ! -f "resnet_152_fft_tiling.py" ] && echo "✗ ERROR: resnet_152_fft_tiling.py not found!" && exit 1

RESULTS_DIR="resnet_152_fft_tiling_results_l40s"
mkdir -p $RESULTS_DIR
cd $RESULTS_DIR

python ../resnet_152_fft_tiling.py
EXIT_CODE=$?

echo "End Time: $(date)"
exit $EXIT_CODE

