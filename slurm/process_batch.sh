#!/bin/bash
# slurm/process_batch.sh
# Process a batch of RAW images
# Usage: sbatch slurm/process_batch.sh <batch_id>
#
# Example: sbatch slurm/process_batch.sh MD_2025-04-14

#SBATCH --job-name=svs_raw
#SBATCH -A dash_agir
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH -o "/project/dash_agir/matthew.kutugata/logs/svs_raw-%j.out"
#SBATCH -e "/project/dash_agir/matthew.kutugata/logs/svs_raw-%j.err"

set -e  # Exit on error

# ========================================
# Configuration
# ========================================
REPO_DIR="$HOME/repos/svs-raw-api"
PROJECT_BASE="/project/dash_agir/matthew.kutugata"
ENV_PATH="$PROJECT_BASE/software/miniforge3/envs/semif_prep"
RAW_DATA_SOURCE="/90daydata/dash_agir/data/semifield-upload"

# Command line arguments
BATCH_ID=${1:?Error: batch_id required. Usage: sbatch process_batch.sh MD_2025-04-14}

# Derived paths
SOURCE_DIR="$RAW_DATA_SOURCE/$BATCH_ID"
TMP_DATA_DIR="/tmp/job_${SLURM_JOB_ID}/data/$BATCH_ID"
TMP_OUTPUT_DIR="/tmp/job_${SLURM_JOB_ID}/output/$BATCH_ID/images"

# ========================================
# Job Info
# ========================================
echo "========================================"
echo "SVS RAW Processing Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Batch ID: $BATCH_ID"
echo "Source: $SOURCE_DIR"
echo "========================================"

# ========================================
# Setup /tmp directories
# ========================================
echo "[1/6] Setting up /tmp workspace..."
mkdir -p "$TMP_DATA_DIR"
mkdir -p "$TMP_OUTPUT_DIR"

echo "Checking /tmp space..."
df -h /tmp | tail -1

# ========================================
# Copy RAW data to /tmp (much faster!)
# ========================================
echo "[2/6] Copying RAW data to /tmp..."
echo "Source: $SOURCE_DIR"
echo "Destination: $TMP_DATA_DIR"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Count files before copying
FILE_COUNT=$(find "$SOURCE_DIR" -name "*.RAW" | wc -l)
echo "Found $FILE_COUNT RAW files to copy"

# Copy with progress
rsync -ah --progress "$SOURCE_DIR/" "$TMP_DATA_DIR/"

echo "✓ Data copied to /tmp"
ls -lh "$TMP_DATA_DIR" | head -10

# ========================================
# Load environment
# ========================================
echo "[3/6] Loading environment..."
module load miniconda
source activate $ENV_PATH

# Source RawTherapee path
source $REPO_DIR/scripts/rawtherapee_path.sh
echo "RawTherapee: $RAWTHERAPEE_CLI"

# ========================================
# Process images from /tmp
# ========================================
echo "[4/6] Processing images from /tmp..."
echo "Input: $TMP_DATA_DIR"
echo "Output: $TMP_OUTPUT_DIR"

# Calculate optimal worker count (aim for 2 threads per image)
WORKERS=$((SLURM_CPUS_PER_TASK / 2))
WORKERS=$(( WORKERS > 0 ? WORKERS : 1 ))

echo "Using $WORKERS parallel workers with $SLURM_CPUS_PER_TASK total CPUs"

python -m svs_raw_api.cli \
    --config "$REPO_DIR/conf/scinet.yaml" \
    --input "$TMP_DATA_DIR" \
    --output "$TMP_OUTPUT_DIR" \
    --batch-id "$BATCH_ID" \
    --threads $SLURM_CPUS_PER_TASK \
    --workers $WORKERS \
    --job-id $SLURM_JOB_ID

PROCESS_EXIT_CODE=$?

# ========================================
# Copy results to permanent storage
# ========================================
if [ $PROCESS_EXIT_CODE -eq 0 ]; then
    echo "[5/6] Processing succeeded. Copying results..."
    
    FINAL_OUTPUT="$PROJECT_BASE/semifield-developed-images/$BATCH_ID/images"
    mkdir -p "$FINAL_OUTPUT"
    
    # Copy with progress (exclude .dng)
    rsync -avh --progress --exclude="*.dng" "$TMP_OUTPUT_DIR/" "$FINAL_OUTPUT/"
    
    # Verify copy
    OUTPUT_FILES=$(find "$FINAL_OUTPUT" -type f | wc -l)
    echo "✓ Results copied to: $FINAL_OUTPUT"
    echo "✓ Files created: $OUTPUT_FILES"
    
    # Quick summary
    echo ""
    echo "Summary:"
    echo "  DNG files: $(find "$FINAL_OUTPUT" -name "*.dng" | wc -l)"
    echo "  JPG files: $(find "$FINAL_OUTPUT" -name "*.jpg" | wc -l)"
else
    echo "ERROR: Processing failed with exit code $PROCESS_EXIT_CODE"
    echo "Check logs at: $PROJECT_BASE/logs/svs_raw-$SLURM_JOB_ID.out"
    exit $PROCESS_EXIT_CODE
fi

# ========================================
# Cleanup /tmp
# ========================================
echo "[6/6] Cleaning up /tmp..."
TMP_JOB_DIR="/tmp/job_${SLURM_JOB_ID}"

# Show space used before cleanup
echo "Space used in /tmp:"
du -sh "$TMP_JOB_DIR" 2>/dev/null || echo "Already cleaned"

# Remove everything
rm -rf "$TMP_JOB_DIR"
echo "✓ Cleaned up /tmp"

# Final /tmp space check
echo "Final /tmp space:"
df -h /tmp | tail -1

echo ""
echo "========================================"
echo "Job completed successfully: $(date)"
echo "Results: $FINAL_OUTPUT"
echo "========================================"