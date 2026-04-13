#!/bin/bash
#SBATCH --job-name=train_all_predictors
#SBATCH --output=train_all_%j.out
#SBATCH --error=train_all_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# ============================================================================
# Train ALL prediction models (layer time, layer energy, model time,
# model energy) using all 4 ML approaches.
#
# Usage (interactive):
#   bash run_train_all.sh
#
# Usage (SLURM):
#   sbatch run_train_all.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

source "$REPO_ROOT/venv/bin/activate"

echo "=============================================="
echo "TRAINING ALL PREDICTORS"
echo "Start: $(date)"
echo "=============================================="

echo ""
echo ">>> 1/5: Layer-level time prediction"
echo ""
python train_layer_time.py

echo ""
echo ">>> 2/5: Layer-level energy prediction"
echo ""
python train_layer_energy.py

echo ""
echo ">>> 3/5: Model-level time prediction"
echo ""
python train_model_time.py

echo ""
echo ">>> 4/5: Model-level energy prediction"
echo ""
python train_model_energy.py

echo ""
echo ">>> 5/5: Unified evaluation"
echo ""
python evaluate.py

echo ""
echo "=============================================="
echo "ALL TRAINING COMPLETE"
echo "End: $(date)"
echo "=============================================="
echo ""
echo "Artifacts saved to: $SCRIPT_DIR/artifacts/"
ls -la "$SCRIPT_DIR/artifacts/"
