#!/bin/bash
# ============================================================================
# Stage 1: Sanity Check
# ============================================================================
# Quick verification that model loads, ai3 converts, and basic timing works.
# Run time: ~5 minutes per combination
#
# Usage:
#   ./1_sanity_check.sh <model> <algorithm> <gpu>
#
# Examples:
#   ./1_sanity_check.sh vgg16 gemm l40s
#   ./1_sanity_check.sh densenet121 direct h100
#   ./1_sanity_check.sh resnet152 fft v100
#
# Check results:
#   cat results/1_sanity/<model>_<algorithm>_*.out
# ============================================================================

set -e

# Parse arguments
MODEL=$1
ALGO=$2
GPU=$3

# Validate arguments
if [ -z "$MODEL" ] || [ -z "$ALGO" ] || [ -z "$GPU" ]; then
    echo "Usage: ./1_sanity_check.sh <model> <algorithm> <gpu>"
    echo ""
    echo "Models: vgg16, densenet121, resnet152, googlenet, alexnet, resnet50"
    echo "Algorithms: direct, gemm, 'implicit gemm', 'implicit precomp gemm',"
    echo "            fft, 'fft tiling', winograd, 'winograd nonfused', guess"
    echo "GPUs: l40s, h100, v100"
    echo ""
    echo "Example: ./1_sanity_check.sh vgg16 gemm l40s"
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

# Create safe algorithm name for filenames (replace spaces with underscores)
ALGO_SAFE="${ALGO// /_}"

# Job configuration
JOB_NAME="SANITY_${MODEL}_${ALGO_SAFE}_${GPU}"
RESULTS_DIR="results/1_sanity/${MODEL}_${ALGO_SAFE}_${GPU}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "STAGE 1: SANITY CHECK"
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
#SBATCH --time=00:15:00
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

# Run profiler in test mode
python run_profile.py \\
    --model "$MODEL" \\
    --algorithm "$ALGO" \\
    --test \\
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
echo ""
