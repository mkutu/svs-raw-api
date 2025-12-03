#!/bin/bash
# ============================================================================
# List Available Batches for Processing
# ============================================================================
#
# Shows batches in scratch storage and their processing status
#
# Usage:
#   ./scripts/list_batches.sh
#
# ============================================================================

SCRATCH_DIR="/90daydata/dash_agir/data/semifield-upload"
OUTPUT_DIR="/project/dash_agir/matthew.kutugata/semifield-developed-images"

echo "============================================================================"
echo "Available Batches"
echo "============================================================================"
echo ""

if [ ! -d "$SCRATCH_DIR" ]; then
    echo "ERROR: Scratch directory not found: $SCRATCH_DIR"
    exit 1
fi

echo "Batches in scratch storage ($SCRATCH_DIR):"
echo ""

printf "%-20s %-10s %-15s\n" "Batch ID" "RAW Files" "Status"
echo "------------------------------------------------------------------------"

for batch_dir in "$SCRATCH_DIR"/*; do
    if [ -d "$batch_dir" ]; then
        batch_id=$(basename "$batch_dir")

        # Count RAW files
        raw_count=$(find "$batch_dir" -maxdepth 1 -name "*.RAW" -o -name "*.ARW" 2>/dev/null | wc -l)

        # Check if processed
        if [ -d "$OUTPUT_DIR/$batch_id/images" ]; then
            jpg_count=$(find "$OUTPUT_DIR/$batch_id/images" -name "*.jpg" 2>/dev/null | wc -l)
            if [ "$jpg_count" -eq "$raw_count" ] && [ "$raw_count" -gt 0 ]; then
                status="✓ Complete"
            elif [ "$jpg_count" -gt 0 ]; then
                status="⚠ Partial ($jpg_count/$raw_count)"
            else
                status="○ Pending"
            fi
        else
            status="○ Pending"
        fi

        printf "%-20s %-10s %-15s\n" "$batch_id" "$raw_count" "$status"
    fi
done

echo ""
echo "Legend:"
echo "  ✓ Complete  - All images processed"
echo "  ⚠ Partial   - Some images processed"
echo "  ○ Pending   - Not yet processed"
echo ""
echo "To process a batch:"
echo "  ./scripts/process_batch.sh <batch_id>"
echo ""
