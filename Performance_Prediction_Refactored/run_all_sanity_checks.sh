#!/bin/bash
# ============================================================================
# Run Sanity Checks for All Combinations
# ============================================================================
# Submits sanity check jobs for all model/algorithm/GPU combinations.
# Automatically skips incompatible combinations (e.g., winograd with non-VGG16).
#
# Usage:
#   ./run_all_sanity_checks.sh
#
# This will submit jobs for all combinations using l40s, h100, and v100 GPUs.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define all options
MODELS=("vgg16" "densenet121" "resnet152" "googlenet" "alexnet" "resnet50")
ALGOS=("direct" "gemm" "implicit gemm" "implicit precomp gemm" "fft" "fft tiling" "winograd" "winograd nonfused" "guess")
GPUS=("l40s" "h100" "v100")

# Check compatibility function
is_compatible() {
    local model=$1
    local algo=$2
    
    # Winograd algorithms only work with vgg16
    if [[ "$algo" == "winograd" ]] || [[ "$algo" == "winograd nonfused" ]]; then
        if [[ "$model" != "vgg16" ]]; then
            return 1  # Not compatible
        fi
    fi
    return 0  # Compatible
}

# Counters
TOTAL=0
SUBMITTED=0
SKIPPED=0

echo "=============================================="
echo "AUTOMATED SANITY CHECK SUBMISSION"
echo "=============================================="
echo "Models:      ${#MODELS[@]} (${MODELS[*]})"
echo "Algorithms:  ${#ALGOS[@]}"
echo "GPUs:        ${#GPUS[@]} (${GPUS[*]})"
echo ""
echo "Total combinations: $((${#MODELS[@]} * ${#ALGOS[@]} * ${#GPUS[@]}))"
echo "=============================================="
echo ""

# Create log file
LOG_FILE="$SCRIPT_DIR/results/1_sanity/submission_log_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$SCRIPT_DIR/results/1_sanity"
echo "Submission log: $LOG_FILE"
echo ""

# Loop through all combinations
for model in "${MODELS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        for gpu in "${GPUS[@]}"; do
            TOTAL=$((TOTAL + 1))
            
            # Check compatibility
            if ! is_compatible "$model" "$algo"; then
                echo "[$TOTAL] ⏭️  SKIP: $model + $algo + $gpu (incompatible)"
                echo "[$TOTAL] SKIP: $model + $algo + $gpu (incompatible)" >> "$LOG_FILE"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            
            # Submit job
            echo "[$TOTAL] 📤 SUBMIT: $model + $algo + $gpu"
            echo "[$TOTAL] SUBMIT: $model + $algo + $gpu" >> "$LOG_FILE"
            
            if "$SCRIPT_DIR/1_sanity_check.sh" "$model" "$algo" "$gpu" >> "$LOG_FILE" 2>&1; then
                SUBMITTED=$((SUBMITTED + 1))
                echo "      ✓ Submitted successfully"
            else
                echo "      ✗ Submission failed (check log)"
            fi
            
            # Small delay to avoid overwhelming SLURM
            sleep 0.5
        done
    done
done

echo ""
echo "=============================================="
echo "SUMMARY"
echo "=============================================="
echo "Total combinations:  $TOTAL"
echo "Successfully submitted: $SUBMITTED"
echo "Skipped (incompatible): $SKIPPED"
echo ""
echo "Check job queue:     squeue -u \$USER"
echo "View submission log: cat $LOG_FILE"
echo ""
