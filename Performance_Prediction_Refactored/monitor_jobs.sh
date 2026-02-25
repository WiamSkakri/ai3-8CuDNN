#!/bin/bash
# ============================================================================
# Live Job Monitor
# ============================================================================
# Continuously monitors job status and updates a live report.
# Runs in the foreground or background, checking every N minutes.
#
# NOTE: This monitor does NOT use sacct (job history), so it won't pick up
#       old test runs. Start this BEFORE submitting jobs for clean tracking.
#
# Status detection (file-based, no history):
#   - RUNNING/PENDING: Job is in squeue (active)
#   - COMPLETED: Result CSV exists (*_overall.csv)
#   - INTERRUPTED: Output file exists (.out) but no results (timeout/failed)
#   - NOT_STARTED: No files yet
#
# Usage:
#   ./monitor_jobs.sh              # Run in foreground (Ctrl+C to stop)
#   ./monitor_jobs.sh &            # Run in background
#   nohup ./monitor_jobs.sh &      # Run in background, survives logout
#
# Options:
#   INTERVAL=300 ./monitor_jobs.sh   # Check every 5 minutes (default: 2 min)
#
# Output:
#   results/3_full/live_status.txt   - Current status (overwritten each check)
#   results/3_full/live_status.csv   - Current status as CSV (overwritten each check)
#   results/3_full/monitor_log.txt   - Historical log of all changes
#
# Stop background monitor:
#   pkill -f monitor_jobs.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/3_full"
LIVE_FILE="$RESULTS_DIR/live_status.txt"
CSV_FILE="$RESULTS_DIR/live_status.csv"
LOG_FILE="$RESULTS_DIR/monitor_log.txt"
INTERVAL=${INTERVAL:-120}  # Default: check every 2 minutes

# Define expected combinations
MODELS=("vgg16" "densenet121" "resnet152" "googlenet")
ALGOS=("direct" "gemm" "implicit_gemm" "implicit_precomp_gemm" "fft" "fft_tiling" "winograd" "winograd_nonfused" "guess")
GPUS=("l40s" "h100" "v100")

mkdir -p "$RESULTS_DIR"

# Check compatibility
is_compatible() {
    local model=$1
    local algo=$2
    if [[ "$algo" == "winograd" ]] || [[ "$algo" == "winograd_nonfused" ]]; then
        [[ "$model" == "vgg16" ]]
    else
        return 0
    fi
}

# Log with timestamp
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

echo "=============================================="
echo "LIVE JOB MONITOR STARTED"
echo "=============================================="
echo "Check interval: ${INTERVAL}s"
echo "Live status:    $LIVE_FILE"
echo "CSV report:     $CSV_FILE"
echo "Monitor log:    $LOG_FILE"
echo ""
echo "Press Ctrl+C to stop (or 'pkill -f monitor_jobs.sh')"
echo "=============================================="
echo ""

log_msg "Monitor started (interval: ${INTERVAL}s)"

# Track previous counts for change detection
PREV_COMPLETED=0
PREV_RUNNING=0
PREV_FAILED=0

while true; do
    # Counters
    COMPLETED=0
    RUNNING=0
    PENDING=0
    FAILED=0        # Interrupted/failed/timeout (has .out but no results)
    NOT_STARTED=0
    
    # Temp file for current check
    {
        echo "=============================================="
        echo "  JOB STATUS - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=============================================="
        echo ""
        echo "MODEL              ALGORITHM                  GPU    STATUS      JOB_ID     ELAPSED"
        echo "------------------ -------------------------- ------ ----------- ---------- --------"
    } > "$LIVE_FILE.tmp"
    
    # CSV header
    echo "timestamp,model,algorithm,gpu,status,job_id,elapsed,has_results" > "$CSV_FILE.tmp"
    
    for model in "${MODELS[@]}"; do
        for algo in "${ALGOS[@]}"; do
            for gpu in "${GPUS[@]}"; do
                
                if ! is_compatible "$model" "$algo"; then
                    continue
                fi
                
                job_name="FULL_${model}_${algo}_${gpu}"
                result_dir="$RESULTS_DIR/${model}_${algo}_${gpu}"
                
                status="NOT_STARTED"
                job_id="-"
                elapsed="-"
                has_results="no"
                
                # Check if results CSV exists
                if ls "$result_dir"/*_overall.csv 2>/dev/null | head -1 > /dev/null; then
                    has_results="yes"
                fi
                
                # Check squeue
                queue_info=$(squeue -u $USER -n "$job_name" -h -o "%i %T %M" 2>/dev/null | head -1)
                
                if [[ -n "$queue_info" ]]; then
                    job_id=$(echo "$queue_info" | awk '{print $1}')
                    state=$(echo "$queue_info" | awk '{print $2}')
                    elapsed=$(echo "$queue_info" | awk '{print $3}')
                    
                    if [[ "$state" == "RUNNING" ]]; then
                        status="RUNNING"
                        ((RUNNING++))
                    elif [[ "$state" == "PENDING" ]]; then
                        status="PENDING"
                        ((PENDING++))
                    else
                        status="$state"
                    fi
                else
                    # Not in queue - check files to determine status
                    # (No sacct - avoids picking up old job history)
                    
                    if [[ "$has_results" == "yes" ]]; then
                        # Has result CSV = completed successfully
                        status="COMPLETED"
                        ((COMPLETED++))
                    elif ls "$result_dir"/*.out 2>/dev/null | head -1 > /dev/null; then
                        # Has .out file but no results = failed/interrupted/timeout
                        status="INTERRUPTED"
                        ((FAILED++))
                    else
                        # No output files at all = not started
                        ((NOT_STARTED++))
                    fi
                fi
                
                printf "%-18s %-26s %-6s %-11s %-10s %s\n" \
                    "$model" "$algo" "$gpu" "$status" "$job_id" "$elapsed" >> "$LIVE_FILE.tmp"
                
                # Write CSV row
                echo "$(date '+%Y-%m-%d %H:%M:%S'),$model,$algo,$gpu,$status,$job_id,$elapsed,$has_results" >> "$CSV_FILE.tmp"
                
            done
        done
    done
    
    TOTAL=$((COMPLETED + RUNNING + PENDING + FAILED + NOT_STARTED))
    
    {
        echo ""
        echo "=============================================="
        echo "                 SUMMARY"
        echo "=============================================="
        printf "  ✅ Completed:    %3d / %d\n" "$COMPLETED" "$TOTAL"
        printf "  🔄 Running:      %3d\n" "$RUNNING"
        printf "  ⏳ Pending:      %3d\n" "$PENDING"
        printf "  ❌ Interrupted:  %3d\n" "$FAILED"
        printf "  ⬜ Not started:  %3d\n" "$NOT_STARTED"
        echo "=============================================="
        echo ""
        echo "Next update in ${INTERVAL}s..."
        echo "View live:  cat $LIVE_FILE"
        echo "View CSV:   cat $CSV_FILE"
        echo "Stop:       pkill -f monitor_jobs.sh"
    } >> "$LIVE_FILE.tmp"
    
    # Atomic move
    mv "$LIVE_FILE.tmp" "$LIVE_FILE"
    mv "$CSV_FILE.tmp" "$CSV_FILE"
    
    # Log changes
    if [[ $COMPLETED -ne $PREV_COMPLETED ]]; then
        log_msg "Completed: $PREV_COMPLETED -> $COMPLETED"
    fi
    if [[ $RUNNING -ne $PREV_RUNNING ]]; then
        log_msg "Running: $PREV_RUNNING -> $RUNNING"
    fi
    if [[ $FAILED -ne $PREV_FAILED ]]; then
        log_msg "Interrupted: $PREV_FAILED -> $FAILED"
    fi
    
    PREV_COMPLETED=$COMPLETED
    PREV_RUNNING=$RUNNING
    PREV_FAILED=$FAILED
    
    # Check if all done
    if [[ $RUNNING -eq 0 ]] && [[ $PENDING -eq 0 ]] && [[ $NOT_STARTED -eq 0 ]]; then
        log_msg "All jobs finished! Completed=$COMPLETED, Interrupted=$FAILED"
        echo ""
        echo "🎉 All jobs finished!"
        cat "$LIVE_FILE"
        break
    fi
    
    # Print current summary to console
    clear 2>/dev/null || true
    cat "$LIVE_FILE"
    
    sleep "$INTERVAL"
done
