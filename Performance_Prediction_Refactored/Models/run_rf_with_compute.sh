#!/bin/bash
#SBATCH --job-name=rf_with_compute
#SBATCH --output=rf_with_compute_%j.out
#SBATCH --error=rf_with_compute_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source activate ai3_8cudnn

cd /home/wxs428/ai3-8CuDNN/models

python rf_with_compute.py
