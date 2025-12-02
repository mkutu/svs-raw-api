#!/bin/bash
# scripts/generate_batch_list.sh
# Generate a list of batch IDs for array job processing
# Usage: bash scripts/generate_batch_list.sh [output_file]

BASE_DIR="/90daydata/dash_agir/data/semifield-upload"
OUTPUT_FILE=${1:-conf/batch_ids.txt}

echo "Scanning for batches in: $BASE_DIR"

# Find all directories matching the pattern MD_* and extract just the directory name
find "$BASE_DIR" -maxdepth 1 -type d -name "MD_*" -printf "%f\n" | sort > "$OUTPUT_FILE"

COUNT=$(wc -l < "$OUTPUT_FILE")
echo "Found $COUNT batches"
echo "Output written to: $OUTPUT_FILE"

echo ""
echo "First 5 batches:"
head -5 "$OUTPUT_FILE"

echo ""
echo "To process all batches:"
echo "  1. Review: cat $OUTPUT_FILE"
echo "  2. Edit array size in slurm/array_job.sh: --array=0-$((COUNT-1))%5"
echo "  3. Submit: sbatch slurm/array_job.sh $OUTPUT_FILE"
echo ""
echo "To process a single batch:"
echo "  sbatch slurm/process_batch.sh $(head -1 $OUTPUT_FILE)"