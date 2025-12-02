#!/bin/bash
# scripts/estimate_tmp_space.sh
# Estimate /tmp space needed for a batch
# Usage: bash scripts/estimate_tmp_space.sh <batch_id>

BATCH_ID=${1:?Error: batch_id required. Usage: bash estimate_tmp_space.sh MD_2025-04-14}
BASE_DIR="/90daydata/dash_agir/data/semifield-upload"
SOURCE_DIR="$BASE_DIR/$BATCH_ID"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Batch directory not found: $SOURCE_DIR"
    exit 1
fi

echo "========================================"
echo "Storage Estimation for: $BATCH_ID"
echo "========================================"
echo ""

# Count files
RAW_COUNT=$(find "$SOURCE_DIR" -name "*.RAW" 2>/dev/null | wc -l)
echo "RAW files: $RAW_COUNT"

if [ $RAW_COUNT -eq 0 ]; then
    echo "No RAW files found in $SOURCE_DIR"
    exit 1
fi

# Get actual size
ACTUAL_SIZE=$(du -sh "$SOURCE_DIR" 2>/dev/null | cut -f1)
echo "Actual batch size: $ACTUAL_SIZE"

# Calculate space needed
INPUT_SIZE_MB=$(du -sm "$SOURCE_DIR" 2>/dev/null | cut -f1)
OUTPUT_SIZE_MB=$((INPUT_SIZE_MB / 2))  # Estimate: outputs ~50% of input size
TOTAL_NEEDED_MB=$((INPUT_SIZE_MB + OUTPUT_SIZE_MB))
TOTAL_NEEDED_GB=$((TOTAL_NEEDED_MB / 1024))

echo ""
echo "--- Space Estimates ---"
echo "Input (RAW):     ${INPUT_SIZE_MB} MB"
echo "Output (DNG/JPG): ${OUTPUT_SIZE_MB} MB (estimated)"
echo "Total needed:    ${TOTAL_NEEDED_MB} MB (~${TOTAL_NEEDED_GB} GB)"
echo ""

# Check typical node /tmp size
echo "--- Recommendation ---"
if [ $TOTAL_NEEDED_GB -lt 50 ]; then
    echo "✓ Small batch - should fit easily on most nodes"
    echo "  Recommended: sbatch slurm/process_batch.sh $BATCH_ID"
elif [ $TOTAL_NEEDED_GB -lt 150 ]; then
    echo "✓ Medium batch - should fit on most nodes"
    echo "  Check node /tmp size: df -h /tmp (typically 200-500GB)"
    echo "  Recommended: sbatch slurm/process_batch.sh $BATCH_ID"
elif [ $TOTAL_NEEDED_GB -lt 300 ]; then
    echo "⚠ Large batch - may not fit on all nodes"
    echo "  Verify node /tmp size before running"
    echo "  Consider splitting into array job if too large"
else
    echo "⚠ Very large batch - likely too big for single node /tmp"
    echo "  Consider:"
    echo "  1. Split batch into smaller chunks"
    echo "  2. Process in multiple array job tasks"
    echo "  3. Request specific nodes with large /tmp"
fi

echo ""
echo "========================================"