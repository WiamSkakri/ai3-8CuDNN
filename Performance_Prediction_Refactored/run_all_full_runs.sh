#!/bin/bash
# ============================================================================
# Run Full Profiling for All Combinations
# ============================================================================
# Submits full run jobs (100 input sizes) for all model/algorithm/GPU combinations.
# Automatically skips incompatible combinations (e.g., winograd with non-VGG16).
#
# Usage:
#   ./run_all_full_runs.sh
#
# This will submit jobs for all combinations using l40s, h100, and v100 GPUs.
#
# WARNING: Full runs take 2-6 hours each. This will submit many long jobs!
#          Total combinations: 6 models × 9 algos × 3 GPUs = 162 jobs
#          (minus incompatible combinations)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define all options
MODELS=("vgg16" "densenet121" "resnet152" "googlenet")
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
echo "AUTOMATED FULL RUN SUBMISSION"
echo "=============================================="
echo "Models:      ${#MODELS[@]} (${MODELS[*]})"
echo "Algorithms:  ${#ALGOS[@]}"
echo "GPUs:        ${#GPUS[@]} (${GPUS[*]})"
echo ""
echo "Total combinations: $((${#MODELS[@]} * ${#ALGOS[@]} * ${#GPUS[@]}))"
echo ""
echo "⚠️  WARNING: Full runs take 2-6 hours EACH!"
echo "=============================================="
echo ""

# Confirmation prompt
read -p "Are you sure you want to submit all full run jobs? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi
echo ""

# Create log file
LOG_FILE="$SCRIPT_DIR/results/3_full/submission_log_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$SCRIPT_DIR/results/3_full"
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
            
            if "$SCRIPT_DIR/3_full_run.sh" "$model" "$algo" "$gpu" >> "$LOG_FILE" 2>&1; then
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
echo "Total combinations:     $TOTAL"
echo "Successfully submitted: $SUBMITTED"
echo "Skipped (incompatible): $SKIPPED"
echo ""
echo "⏱️  Estimated total GPU-hours: $((SUBMITTED * 4)) - $((SUBMITTED * 6)) hours"
echo ""
echo "Check job queue:     squeue -u \$USER"
echo "View submission log: cat $LOG_FILE"
echo ""
