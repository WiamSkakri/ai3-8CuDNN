#!/bin/bash
# ============================================================================
# Short Slurm test for v2 profiler (--test: 1 input size, 2 measure iters).
#
# Usage:
#   ./run_profile_test.sh <model> <algorithm> <gpu>
#
# Examples:
#   ./run_profile_test.sh vgg16 gemm l40s
#   ./run_profile_test.sh densenet121 "fft tiling" h100
# ============================================================================

set -e

MODEL=$1
ALGO=$2
GPU=$3

if [ -z "$MODEL" ] || [ -z "$ALGO" ] || [ -z "$GPU" ]; then
    echo "Usage: ./run_profile_test.sh <model> <algorithm> <gpu>"
    echo ""
    echo "Models: vgg16, densenet121, resnet152, googlenet, alexnet, resnet50"
    echo "Algorithms: direct, gemm, 'implicit gemm', 'implicit precomp gemm',"
    echo "            fft, 'fft tiling', winograd, 'winograd nonfused', guess"
    echo "GPUs: l40s, h100, v100, 2080, 4090"
    echo ""
    echo "Example: ./run_profile_test.sh vgg16 gemm l40s"
    exit 1
fi

case $GPU in
    l40s)   CONSTRAINT="gpul40s" ;;
    h100)   CONSTRAINT="gpu2h100" ;;
    v100)   CONSTRAINT="gpu4v100" ;;
    2080)   CONSTRAINT="gpu2080" ;;
    4090)   CONSTRAINT="gpu4090" ;;
    *)
        echo "Unknown GPU: $GPU"
        echo "  Use: l40s, h100, v100, 2080, or 4090"
        exit 1
        ;;
esac

ALGO_SAFE="${ALGO// /_}"
JOB_NAME="test_${MODEL}_${ALGO_SAFE}_${GPU}"
RESULTS_DIR="results/test_${MODEL}_${ALGO_SAFE}_${GPU}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================="
echo "v2 PROFILER: TEST (--test)"
echo "=============================================="
echo "Model:      $MODEL"
echo "Algorithm:  $ALGO"
echo "GPU:        $GPU ($CONSTRAINT)"
echo "Results:    $RESULTS_DIR"
echo "=============================================="

mkdir -p "$SCRIPT_DIR/$RESULTS_DIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$SCRIPT_DIR/$RESULTS_DIR/%x_%j.out
#SBATCH --error=$SCRIPT_DIR/$RESULTS_DIR/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -C $CONSTRAINT

echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Model: $MODEL"
echo "Algorithm: $ALGO"
echo "GPU: $CONSTRAINT"
echo "Start: \$(date)"

nvidia-smi

source "$REPO_ROOT/venv/bin/activate"

cd "$SCRIPT_DIR"

python run_profile.py \\
    --model "$MODEL" \\
    --algorithm "$ALGO" \\
    --test \\
    --output-dir "$RESULTS_DIR"

echo "End: \$(date)"
echo "Exit: \$?"
EOF

echo ""
echo "Submitted: $JOB_NAME"
echo "Check: squeue -u \$USER"
echo "Outputs: $SCRIPT_DIR/$RESULTS_DIR/"
