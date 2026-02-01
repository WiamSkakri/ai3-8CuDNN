#!/bin/bash
# ============================================================================
# Stage 2: Quick Sweep
# ============================================================================
# Test 10 input sizes to verify scaling behavior looks reasonable.
# Run time: ~30-60 minutes per combination
#
# Usage:
#   ./2_quick_sweep.sh <model> <algorithm> <gpu>
#
# Examples:
#   ./2_quick_sweep.sh vgg16 gemm l40s
#   ./2_quick_sweep.sh densenet121 direct h100
#
# Check results:
#   cat results/2_quick/<model>_<algorithm>_*.out
#   head results/2_quick/<model>_<algorithm>_<gpu>/*_overall.csv
# ============================================================================

set -e

# Parse arguments
MODEL=$1
ALGO=$2
GPU=$3

# Validate arguments
if [ -z "$MODEL" ] || [ -z "$ALGO" ] || [ -z "$GPU" ]; then
    echo "Usage: ./2_quick_sweep.sh <model> <algorithm> <gpu>"
    echo ""
    echo "Models: vgg16, densenet121, resnet152, googlenet, alexnet, resnet50"
    echo "Algorithms: direct, gemm, 'implicit gemm', 'implicit precomp gemm',"
    echo "            fft, 'fft tiling', winograd, 'winograd nonfused', guess"
    echo "GPUs: l40s, h100, v100"
    echo ""
    echo "Example: ./2_quick_sweep.sh vgg16 gemm l40s"
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
JOB_NAME="QUICK_${MODEL}_${ALGO_SAFE}_${GPU}"
RESULTS_DIR="results/2_quick/${MODEL}_${ALGO_SAFE}_${GPU}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "STAGE 2: QUICK SWEEP (10 input sizes)"
echo "=============================================="
echo "Model:      $MODEL"
echo "Algorithm:  $ALGO"
echo "GPU:        $GPU ($CONSTRAINT)"
echo "Results:    $RESULTS_DIR"
echo "=============================================="

# Create results directory
mkdir -p "$SCRIPT_DIR/$RESULTS_DIR"

# Submit job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$SCRIPT_DIR/$RESULTS_DIR/%x_%j.out
#SBATCH --error=$SCRIPT_DIR/$RESULTS_DIR/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

# Run profiler in quick mode (10 sizes)
python run_profile.py \\
    --model "$MODEL" \\
    --algorithm "$ALGO" \\
    --quick \\
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
