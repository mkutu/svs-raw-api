#!/bin/bash
# ============================================================================
# Quick Batch Processing Script
# ============================================================================
#
# Convenient wrapper to submit a batch for processing
#
# Usage:
#   ./scripts/process_batch.sh <batch_id>
#
# Example:
#   ./scripts/process_batch.sh MD_2025-10-22
#
# ============================================================================

set -e

BATCH_ID=${1:?Error: batch_id required. Usage: ./scripts/process_batch.sh MD_2025-10-22}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================================"
echo "SVS RAW Processing Pipeline"
echo "============================================================================"
echo "Batch ID: $BATCH_ID"
echo ""

# Check if batch exists
SCRATCH_DIR="/90daydata/dash_agir/data/semifield-upload"
if [ ! -d "$SCRATCH_DIR/$BATCH_ID" ]; then
    echo "ERROR: Batch not found in scratch storage: $SCRATCH_DIR/$BATCH_ID"
    echo ""
    echo "Please transfer the batch to Ceres first using Globus or rsync"
    exit 1
fi

# Count files
RAW_COUNT=$(find "$SCRATCH_DIR/$BATCH_ID" -name "*.RAW" -o -name "*.ARW" | wc -l)
echo "Found $RAW_COUNT RAW files"
echo ""

# Submit to SLURM
echo "Submitting to SLURM queue..."
cd "$REPO_DIR"
JOB_ID=$(sbatch --parsable slurm/submit_snakemake.sh "$BATCH_ID")

echo "✓ Job submitted: $JOB_ID"
echo ""
echo "Monitor progress:"
echo "  squeue -j $JOB_ID"
echo "  tail -f /project/dash_agir/matthew.kutugata/logs/snakemake_${JOB_ID}.out"
echo ""
echo "============================================================================"
