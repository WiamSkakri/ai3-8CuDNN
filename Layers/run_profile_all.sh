#!/bin/bash
# ============================================================================
# Submit v2 profiling jobs for ALL model/algorithm/GPU combinations.
#
# Usage:
#   ./run_profile_all.sh          # submit all 62+ combos
#   ./run_profile_all.sh l40s     # submit all combos for one GPU only
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_FILTER=${1:-"all"}

MODELS=("vgg16" "densenet121" "resnet152" "googlenet")
ALGORITHMS=("gemm" "implicit gemm" "implicit precomp gemm" "fft tiling" "guess")
GPUS=("l40s" "h100" "v100")

# Winograd only works with vgg16
VGG_EXTRA=("winograd" "winograd nonfused")

COUNT=0

for gpu in "${GPUS[@]}"; do
    if [ "$GPU_FILTER" != "all" ] && [ "$GPU_FILTER" != "$gpu" ]; then
        continue
    fi

    for model in "${MODELS[@]}"; do
        for algo in "${ALGORITHMS[@]}"; do
            echo "Submitting: $model + $algo + $gpu"
            "$SCRIPT_DIR/run_profile_full.sh" "$model" "$algo" "$gpu"
            COUNT=$((COUNT + 1))
            sleep 1
        done

        # VGG16-only algorithms
        if [ "$model" = "vgg16" ]; then
            for algo in "${VGG_EXTRA[@]}"; do
                echo "Submitting: $model + $algo + $gpu"
                "$SCRIPT_DIR/run_profile_full.sh" "$model" "$algo" "$gpu"
                COUNT=$((COUNT + 1))
                sleep 1
            done
        fi
    done
done

echo ""
echo "=============================================="
echo "Submitted $COUNT profiling jobs"
echo "Check status: squeue -u \$USER"
echo "=============================================="
