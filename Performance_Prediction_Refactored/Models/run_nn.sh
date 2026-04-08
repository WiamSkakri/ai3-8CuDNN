#!/bin/bash
#SBATCH --job-name=nn
#SBATCH --output=nn_%j.out
#SBATCH --error=nn_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source activate ai3_8cudnn

cd /home/wxs428/ai3-8CuDNN/models

python nn.py
