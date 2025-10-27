#!/bin/bash
#SBATCH --job-name=resnet_152_direct_h100
#SBATCH --output=resnet_152_direct_results_h100/resnet_152_direct_h100_%j.out
#SBATCH --error=resnet_152_direct_results_h100/resnet_152_direct_h100_%j.err
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
[ ! -f "resnet_152_direct.py" ] && echo "✗ ERROR: resnet_152_direct.py not found!" && exit 1

RESULTS_DIR="resnet_152_direct_results_h100"
mkdir -p $RESULTS_DIR
cd $RESULTS_DIR

python ../resnet_152_direct.py
EXIT_CODE=$?

echo "End Time: $(date)"
exit $EXIT_CODE

