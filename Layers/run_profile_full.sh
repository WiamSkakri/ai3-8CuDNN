#!/bin/bash
# ============================================================================
# Full profiling run with v2 profiler (per-layer power/energy).
#
# Usage:
#   ./run_profile_full.sh <model> <algorithm> <gpu>
#
# Examples:
#   ./run_profile_full.sh vgg16 gemm l40s
#   ./run_profile_full.sh densenet121 "implicit gemm" h100
# ============================================================================

set -e

MODEL=$1
ALGO=$2
GPU=$3

if [ -z "$MODEL" ] || [ -z "$ALGO" ] || [ -z "$GPU" ]; then
    echo "Usage: ./run_profile_full.sh <model> <algorithm> <gpu>"
    echo ""
    echo "Models: vgg16, densenet121, resnet152, googlenet, alexnet, resnet50"
    echo "Algorithms: direct, gemm, 'implicit gemm', 'implicit precomp gemm',"
    echo "            fft, 'fft tiling', winograd, 'winograd nonfused', guess"
    echo "GPUs: l40s, h100, v100"
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
        exit 1
        ;;
esac

ALGO_SAFE="${ALGO// /_}"
JOB_NAME="v2_${MODEL}_${ALGO_SAFE}_${GPU}"
RESULTS_DIR="results/${MODEL}_${ALGO_SAFE}_${GPU}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================="
echo "v2 PROFILER: FULL RUN (100 input sizes)"
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
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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

cd $SCRIPT_DIR

python run_profile.py \\
    --model "$MODEL" \\
    --algorithm "$ALGO" \\
    --num-sizes 100 \\
    --warmup 10 \\
    --measure 20 \\
    --output-dir "$RESULTS_DIR"

echo "End: \$(date)"
echo "Exit: \$?"
EOF

echo ""
echo "Job submitted: $JOB_NAME"
echo "Check: squeue -u \$USER"
