#!/bin/bash
# slurm/array_job.sh
# Process multiple batches in parallel using SLURM array jobs
# Usage: 
#   1. Create a file listing all batch IDs (one per line)
#   2. sbatch slurm/array_job.sh batch_ids.txt

#SBATCH --job-name=svs_array
#SBATCH -A dash_agir
#SBATCH -p short
#SBATCH -n=28                       # Number of cores
#SBATCH -N=1                       # Number of nodes
#SBATCH -t=02:00:00            # Max time per task
#SBATCH -o "/project/dash_agir/matthew.kutugata/logs/svs_array-%A_%a.out"
#SBATCH -e "/project/dash_agir/matthew.kutugata/logs/svs_array-%A_%a.err"

set -e

# ========================================
# Configuration
# ========================================
REPO_DIR="$HOME/repos/svs-raw-api"
PROJECT_BASE="/project/dash_agir/matthew.kutugata"
ENV_PATH="$PROJECT_BASE/software/miniforge3/envs/semif_prep"
RAW_DATA_SOURCE="/90daydata/dash_agir/data/semifield-upload"

# Input file listing batch IDs to process
BATCH_LIST=${1:-$REPO_DIR/conf/batch_ids.txt}

# ========================================
# Get this task's batch ID
# ========================================
BATCH_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $BATCH_LIST)

if [ -z "$BATCH_ID" ]; then
    echo "ERROR: No batch ID for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Setup paths
SOURCE_DIR="$RAW_DATA_SOURCE/$BATCH_ID"
TMP_DATA_DIR="/tmp/array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/data/$BATCH_ID"
TMP_OUTPUT_DIR="/tmp/array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/output/$BATCH_ID"

# ========================================
# Job Info
# ========================================
echo "========================================"
echo "Array Job: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Batch ID: $BATCH_ID"
echo "========================================"

# ========================================
# Setup /tmp and copy data
# ========================================
echo "[1/5] Copying data to /tmp..."
mkdir -p "$TMP_DATA_DIR"
mkdir -p "$TMP_OUTPUT_DIR"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

rsync -ah --progress "$SOURCE_DIR/" "$TMP_DATA_DIR/"
FILE_COUNT=$(find "$TMP_DATA_DIR" -name "*.RAW" | wc -l)
echo "✓ Copied $FILE_COUNT RAW files to /tmp"

# ========================================
# Load environment
# ========================================
echo "[2/5] Loading environment..."
module load miniconda
source activate $ENV_PATH
source $REPO_DIR/scripts/rawtherapee_path.sh

# ========================================
# Process this batch from /tmp
# ========================================
echo "[3/5] Processing batch $BATCH_ID..."

# Calculate optimal worker count
WORKERS=$((SLURM_CPUS_PER_TASK / 2))
WORKERS=$(( WORKERS > 0 ? WORKERS : 1 ))

python -m svs_raw_api.cli \
    --config "$REPO_DIR/conf/scinet.yaml" \
    --input "$TMP_DATA_DIR" \
    --output "$TMP_OUTPUT_DIR" \
    --batch-id "$BATCH_ID" \
    --threads $SLURM_CPUS_PER_TASK \
    --workers $WORKERS \
    --job-id "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

# ========================================
# Copy results
# ========================================
echo "[4/5] Copying results to permanent storage..."
if [ $? -eq 0 ]; then
    FINAL_OUTPUT="$PROJECT_BASE/semifield-developed-images/$BATCH_ID"
    mkdir -p "$FINAL_OUTPUT"
    rsync -avh "$TMP_OUTPUT_DIR/" "$FINAL_OUTPUT/"
    
    OUTPUT_FILES=$(find "$FINAL_OUTPUT" -type f | wc -l)
    echo "✓ Results copied: $OUTPUT_FILES files"
else
    echo "ERROR: Processing failed for $BATCH_ID"
    exit 1
fi

# ========================================
# Cleanup /tmp
# ========================================
echo "[5/5] Cleaning up /tmp..."
TMP_JOB_DIR="/tmp/array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
rm -rf "$TMP_JOB_DIR"
echo "✓ Task $SLURM_ARRAY_TASK_ID complete"