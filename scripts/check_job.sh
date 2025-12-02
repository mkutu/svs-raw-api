#!/bin/bash
# scripts/check_job.sh
# Quick diagnostics for a running or completed job
# Usage: bash scripts/check_job.sh <job_id>

JOB_ID=${1:?Error: job_id required. Usage: bash check_job.sh 12345}

echo "========================================"
echo "Job Diagnostics: $JOB_ID"
echo "========================================"
echo ""

# Job status
echo "--- Job Status ---"
squeue -j $JOB_ID 2>/dev/null || echo "Job not in queue (completed or not found)"
echo ""

# Job accounting
echo "--- Resource Usage ---"
sacct -j $JOB_ID --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize,NodeList -P | column -t -s'|'
echo ""

# Check logs
PROJECT_BASE="/project/dash_agir/matthew.kutugata"
OUT_LOG="$PROJECT_BASE/logs/svs_raw-${JOB_ID}.out"
ERR_LOG="$PROJECT_BASE/logs/svs_raw-${JOB_ID}.err"

echo "--- Log Files ---"
if [ -f "$OUT_LOG" ]; then
    echo "Output log: $OUT_LOG"
    echo "Last 10 lines:"
    tail -10 "$OUT_LOG"
else
    echo "Output log not found: $OUT_LOG"
fi
echo ""

if [ -f "$ERR_LOG" ] && [ -s "$ERR_LOG" ]; then
    echo "Error log: $ERR_LOG"
    echo "Last 10 lines:"
    tail -10 "$ERR_LOG"
else
    echo "No errors in log"
fi
echo ""

# Check for output files
echo "--- Output Files ---"
OUTPUT_PATTERN="$PROJECT_BASE/semifield-developed-images/MD_*"
RECENT_OUTPUTS=$(find $OUTPUT_PATTERN -type f -mmin -120 2>/dev/null | wc -l)
echo "Files created in last 2 hours: $RECENT_OUTPUTS"

if [ $RECENT_OUTPUTS -gt 0 ]; then
    echo "Most recent output directory:"
    find $OUTPUT_PATTERN -type f -mmin -120 -printf '%h\n' 2>/dev/null | sort -u | head -1
fi
echo ""

# Check if job is still running - offer /tmp check
RUNNING=$(squeue -j $JOB_ID -h 2>/dev/null | wc -l)
if [ $RUNNING -gt 0 ]; then
    NODE=$(squeue -j $JOB_ID -h -o "%N" 2>/dev/null)
    echo "--- Node Information ---"
    echo "Running on node: $NODE"
    echo ""
    echo "To check /tmp space on the node:"
    echo "  ssh $NODE 'df -h /tmp'"
    echo ""
    echo "To check for job directories:"
    echo "  ssh $NODE 'ls -lh /tmp/job_* /tmp/array_* 2>/dev/null'"
fi

echo "========================================"