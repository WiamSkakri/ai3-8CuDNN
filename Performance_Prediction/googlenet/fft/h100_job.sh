#!/bin/bash
#SBATCH --job-name=googlenet_fft_h100
#SBATCH --output=googlenet_fft_results_h100/googlenet_fft_h100_%j.out
#SBATCH --error=googlenet_fft_results_h100/googlenet_fft_h100_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH -C gpu2h100
#SBATCH --gres=gpu:1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "=========================================="
nvidia-smi

source ~/.bashrc
conda activate ai3_8cudnn

cd $SLURM_SUBMIT_DIR
[ ! -f "googlenet_fft.py" ] && echo "✗ ERROR: googlenet_fft.py not found!" && exit 1

RESULTS_DIR="googlenet_fft_results_h100"
mkdir -p $RESULTS_DIR
cd $RESULTS_DIR

python ../googlenet_fft.py
EXIT_CODE=$?

echo "End Time: $(date)"
exit $EXIT_CODE
