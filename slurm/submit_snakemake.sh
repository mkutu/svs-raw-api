#!/bin/bash
#SBATCH --job-name=svs_snakemake
#SBATCH -A dash_agir
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=08:00:00
#SBATCH -o "/project/dash_agir/matthew.kutugata/logs/snakemake_%j.out"
#SBATCH -e "/project/dash_agir/matthew.kutugata/logs/snakemake_%j.err"

# ============================================================================
# SVS RAW Processing Pipeline - Snakemake Submission Script
# ============================================================================
#
# This script submits the Snakemake workflow to process a batch of RAW images
# The workflow itself will spawn individual SLURM jobs for each image
#
# Usage:
#   sbatch slurm/submit_snakemake.sh <batch_id>
#
# Example:
#   sbatch slurm/submit_snakemake.sh MD_2025-10-22
#
# ============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Configuration
# ============================================================================

BATCH_ID=${1:?Error: batch_id required. Usage: sbatch submit_snakemake.sh MD_2025-10-22}
REPO_DIR="$HOME/repos/svs-raw-api"
ENV_PATH="/project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep"
SCRATCH_DIR="/90daydata/dash_agir/data/semifield-upload"
OUTPUT_DIR="/project/dash_agir/matthew.kutugata/semifield-developed-images"

# ============================================================================
# Print Job Information
# ============================================================================

echo "============================================================================"
echo "SVS RAW Processing Pipeline - Snakemake Workflow"
echo "============================================================================"
echo "Batch ID:       $BATCH_ID"
echo "SLURM Job ID:   $SLURM_JOB_ID"
echo "Node:           $SLURM_NODELIST"
echo "Started:        $(date)"
echo "============================================================================"
echo ""

# ============================================================================
# Validate Input
# ============================================================================

INPUT_PATH="$SCRATCH_DIR/$BATCH_ID"
if [ ! -d "$INPUT_PATH" ]; then
    echo "ERROR: Batch directory not found: $INPUT_PATH"
    echo "Please ensure the batch has been transferred to Ceres scratch storage"
    exit 1
fi

# Count RAW files
RAW_COUNT=$(find "$INPUT_PATH" -name "*.RAW" -o -name "*.ARW" | wc -l)
if [ "$RAW_COUNT" -eq 0 ]; then
    echo "ERROR: No RAW files found in $INPUT_PATH"
    exit 1
fi

echo "Found $RAW_COUNT RAW files to process"
echo ""

# ============================================================================
# Setup Environment
# ============================================================================

echo "Setting up environment..."

# Load conda
module load miniconda 2>/dev/null || true

# Activate environment
if [ -d "$ENV_PATH" ]; then
    source "$ENV_PATH/bin/activate"
    echo "✓ Activated conda environment: semif_prep"
else
    echo "WARNING: Conda environment not found at $ENV_PATH"
    echo "Attempting to use system Python..."
fi

# Verify Snakemake is available
if ! command -v snakemake &> /dev/null; then
    echo "ERROR: Snakemake not found"
    echo "Please install: pip install snakemake"
    exit 1
fi

echo "✓ Snakemake version: $(snakemake --version)"

# Check for RawTherapee
if [ -f "$REPO_DIR/scripts/rawtherapee_path.sh" ]; then
    source "$REPO_DIR/scripts/rawtherapee_path.sh"
    echo "✓ RawTherapee CLI: $RAWTHERAPEE_CLI"
fi

# ============================================================================
# Change to Repository Directory
# ============================================================================

cd "$REPO_DIR"
echo "✓ Working directory: $(pwd)"
echo ""

# ============================================================================
# Run Snakemake
# ============================================================================

echo "============================================================================"
echo "Launching Snakemake Pipeline"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  - Max parallel jobs: 12"
echo "  - Resources per job: 4 CPUs, 16GB RAM"
echo "  - Total max resources: 48 CPUs, 192GB RAM"
echo "  - Profile: config/slurm"
echo ""
echo "Output will be saved to:"
echo "  $OUTPUT_DIR/$BATCH_ID/"
echo ""

# Run Snakemake with SLURM profile
snakemake \
    --profile config/slurm \
    --config batch_id="$BATCH_ID" \
    --verbose \
    --printshellcmds \
    --reason

SNAKE_EXIT=$?

# ============================================================================
# Report Results
# ============================================================================

echo ""
echo "============================================================================"
if [ $SNAKE_EXIT -eq 0 ]; then
    echo "Pipeline Completed Successfully"
    echo "============================================================================"
    echo "Batch ID:       $BATCH_ID"
    echo "Output:         $OUTPUT_DIR/$BATCH_ID/"
    echo "Completed:      $(date)"
    echo ""

    # Show summary if it exists
    SUMMARY="$OUTPUT_DIR/$BATCH_ID/processing_summary.txt"
    if [ -f "$SUMMARY" ]; then
        echo "Processing Summary:"
        echo "-------------------"
        cat "$SUMMARY"
    fi
else
    echo "Pipeline Failed"
    echo "============================================================================"
    echo "Batch ID:       $BATCH_ID"
    echo "Exit code:      $SNAKE_EXIT"
    echo "Failed:         $(date)"
    echo ""
    echo "Check logs at:"
    echo "  $OUTPUT_DIR/$BATCH_ID/logs/"
    echo "  /project/dash_agir/matthew.kutugata/logs/snakemake_$SLURM_JOB_ID.err"
fi

echo "============================================================================"
echo ""

exit $SNAKE_EXIT
