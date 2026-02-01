#!/bin/bash
# ============================================================================
# Check Profiling Status
# ============================================================================
# Shows progress across all three stages and currently running jobs.
#
# Usage:
#   ./check_status.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "PROFILING STATUS"
echo "=============================================="

# Count completed runs per stage
echo ""
echo "📋 STAGE 1 - Sanity Checks:"
if [ -d "results/1_sanity" ]; then
    SANITY_COUNT=$(find results/1_sanity -name "*_overall.csv" 2>/dev/null | wc -l | tr -d ' ')
    echo "   Completed: $SANITY_COUNT combinations"
    if [ "$SANITY_COUNT" -gt 0 ]; then
        echo "   Combinations:"
        for dir in results/1_sanity/*/; do
            if [ -d "$dir" ]; then
                basename "$dir" | sed 's/^/     - /'
            fi
        done
    fi
else
    echo "   No results yet"
fi

echo ""
echo "🏃 STAGE 2 - Quick Sweeps (10 sizes):"
if [ -d "results/2_quick" ]; then
    QUICK_COUNT=$(find results/2_quick -name "*_overall.csv" 2>/dev/null | wc -l | tr -d ' ')
    echo "   Completed: $QUICK_COUNT combinations"
    if [ "$QUICK_COUNT" -gt 0 ]; then
        echo "   Combinations:"
        for dir in results/2_quick/*/; do
            if [ -d "$dir" ]; then
                basename "$dir" | sed 's/^/     - /'
            fi
        done
    fi
else
    echo "   No results yet"
fi

echo ""
echo "📊 STAGE 3 - Full Runs (100 sizes):"
if [ -d "results/3_full" ]; then
    FULL_COUNT=$(find results/3_full -name "*_overall.csv" 2>/dev/null | wc -l | tr -d ' ')
    echo "   Completed: $FULL_COUNT combinations"
    if [ "$FULL_COUNT" -gt 0 ]; then
        echo "   Combinations:"
        for dir in results/3_full/*/; do
            if [ -d "$dir" ]; then
                basename "$dir" | sed 's/^/     - /'
            fi
        done
    fi
else
    echo "   No results yet"
fi

echo ""
echo "=============================================="
echo "RUNNING JOBS"
echo "=============================================="
echo ""

# Check if squeue is available (we're on HPC)
if command -v squeue &> /dev/null; then
    JOBS=$(squeue -u $USER -o "%.10i %.25j %.8T %.12M %.6D %R" 2>/dev/null | grep -E "SANITY|QUICK|FULL" || true)
    if [ -n "$JOBS" ]; then
        echo "JobID      Name                      State    Time         Nodes  Reason"
        echo "$JOBS"
    else
        echo "No profiling jobs currently running."
    fi
else
    echo "(squeue not available - not on HPC login node)"
fi

echo ""
echo "=============================================="
echo "QUICK COMMANDS"
echo "=============================================="
echo ""
echo "Submit sanity check:  ./1_sanity_check.sh <model> <algo> <gpu>"
echo "Submit quick sweep:   ./2_quick_sweep.sh <model> <algo> <gpu>"
echo "Submit full run:      ./3_full_run.sh <model> <algo> <gpu>"
echo ""
echo "View job queue:       squeue -u \$USER"
echo "Cancel a job:         scancel <job_id>"
echo "Cancel all my jobs:   scancel -u \$USER"
echo ""
