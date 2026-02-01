#!/bin/bash
# ============================================================================
# Stage 3: Full Run
# ============================================================================
# Complete profiling with 100 input sizes for final data collection.
# Run time: ~2-6 hours per combination
#
# Usage:
#   ./3_full_run.sh <model> <algorithm> <gpu>
#
# Examples:
#   ./3_full_run.sh vgg16 gemm l40s
#   ./3_full_run.sh densenet121 direct h100
#
# Check results:
#   cat results/3_full/<model>_<algorithm>_*.out
#   head results/3_full/<model>_<algorithm>_<gpu>/*_overall.csv
# ============================================================================

set -e

# Parse arguments
MODEL=$1
ALGO=$2
GPU=$3

# Validate arguments
if [ -z "$MODEL" ] || [ -z "$ALGO" ] || [ -z "$GPU" ]; then
    echo "Usage: ./3_full_run.sh <model> <algorithm> <gpu>"
    echo ""
    echo "Models: vgg16, densenet121, resnet152, googlenet, alexnet, resnet50"
    echo "Algorithms: direct, gemm, 'implicit gemm', 'implicit precomp gemm',"
    echo "            fft, 'fft tiling', winograd, 'winograd nonfused', guess"
    echo "GPUs: l40s, h100, v100"
    echo ""
    echo "Example: ./3_full_run.sh vgg16 gemm l40s"
    exit 1
fi

# Map GPU to SLURM constraint
case $GPU in
    l40s)   CONSTRAINT="gpul40s" ;;
    h100)   CONSTRAINT="gpu2h100" ;;
    v100)   CONSTRAINT="gpu4v100" ;;
    2080)   CONSTRAINT="gpu2080" ;;
    4090)   CONSTRAINT="gpu4090" ;;
    *)      
        echo "✗ Unknown GPU: $GPU"
        echo "  Available: l40s, h100, v100, 2080, 4090"
        exit 1 
        ;;
esac

# Create safe algorithm name for filenames
ALGO_SAFE="${ALGO// /_}"

# Job configuration
JOB_NAME="FULL_${MODEL}_${ALGO_SAFE}_${GPU}"
RESULTS_DIR="results/3_full/${MODEL}_${ALGO_SAFE}_${GPU}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "STAGE 3: FULL RUN (100 input sizes)"
echo "=============================================="
echo "Model:      $MODEL"
echo "Algorithm:  $ALGO"
echo "GPU:        $GPU ($CONSTRAINT)"
echo "Results:    $RESULTS_DIR"
echo "Time Limit: 24 hours"
echo "=============================================="

# Create results directory
mkdir -p "$SCRIPT_DIR/$RESULTS_DIR"

# Submit job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$SCRIPT_DIR/$RESULTS_DIR/%x_%j.out
#SBATCH --error=$SCRIPT_DIR/$RESULTS_DIR/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -C $CONSTRAINT

echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Model: $MODEL"
echo "Algorithm: $ALGO"
echo "GPU Constraint: $CONSTRAINT"
echo "Start Time: \$(date)"
echo "=========================================="

nvidia-smi

# Activate environment
source ~/.bashrc
conda activate ai3_8cudnn

# Change to script directory
cd $SCRIPT_DIR

# Run profiler with full 100 sizes
python run_profile.py \\
    --model "$MODEL" \\
    --algorithm "$ALGO" \\
    --num-sizes 100 \\
    --warmup 10 \\
    --measure 20 \\
    --output-dir "$RESULTS_DIR"

EXIT_CODE=\$?

echo "=========================================="
echo "End Time: \$(date)"
echo "Exit Code: \$EXIT_CODE"
echo "=========================================="

exit \$EXIT_CODE
EOF

echo ""
echo "✓ Job submitted: $JOB_NAME"
echo ""
echo "Check status:  squeue -u \$USER"
echo "View output:   cat $RESULTS_DIR/${JOB_NAME}_*.out"
echo "View CSV:      head $RESULTS_DIR/*_overall.csv"
echo ""
