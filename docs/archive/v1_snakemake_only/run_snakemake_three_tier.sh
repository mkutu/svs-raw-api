#!/bin/bash
#SBATCH --job-name=svs_pipeline_3tier
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/project/dash_agir/matthew.kutugata/logs/snakemake_%x_%j.out
#SBATCH --error=/project/dash_agir/matthew.kutugata/logs/snakemake_%x_%j.err

#
# Three-Tier SVS RAW Image Processing Pipeline
# Submits Snakemake workflow that spawns parallel SLURM jobs
#
# Usage:
#   sbatch -A dash_agir run_snakemake_three_tier.sh <BATCH_ID> [CONFIG_FILE]
#

set -e

# Get batch ID from command line
BATCH_ID="${1:-}"
CONFIG_FILE="${2:-config/snakemake_config_three_tier.yaml}"

# Configuration
REPO_ROOT="$HOME/repos/svs-raw-api"
CONDA_ENV="/project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep"
DB_PATH="/project/dash_agir/matthew.kutugata/pipeline_tracking.db"
SNAKEFILE="Snakefile_three_tier"

# Validation
if [ -z "$BATCH_ID" ]; then
    echo "❌ Error: BATCH_ID required"
    echo "Usage: sbatch -A dash_agir $0 <BATCH_ID> [CONFIG_FILE]"
    exit 1
fi

if [ ! -f "$REPO_ROOT/$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $REPO_ROOT/$CONFIG_FILE"
    exit 1
fi

# Print job info
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Three-Tier SVS RAW Processing Pipeline                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID:       $SLURM_JOB_ID"
echo "Batch ID:     $BATCH_ID"
echo "Node:         $SLURM_NODELIST"
echo "Config:       $CONFIG_FILE"
echo "Started:      $(date)"
echo ""

# Check database
if [ ! -f "$DB_PATH" ]; then
    echo "⚠️  Warning: Database not found at $DB_PATH"
    echo "   Database updates will be skipped"
else
    echo "✓ Database:    $DB_PATH"
fi

# Update database: Set processing status to 'processing'
if [ -f "$DB_PATH" ]; then
    sqlite3 "$DB_PATH" <<EOF
UPDATE batches 
SET processing_status = 'processing',
    processing_started_at = datetime('now'),
    updated_at = datetime('now')
WHERE batch_id = '$BATCH_ID';

INSERT INTO processing_history (batch_id, job_id, status, started_at)
VALUES ('$BATCH_ID', '$SLURM_JOB_ID', 'started', datetime('now'));
EOF
    echo "✓ Database status updated to 'processing'"
fi

# Activate conda environment
echo ""
echo "Activating conda environment..."
source "$CONDA_ENV/bin/activate"

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate conda environment"
    exit 1
fi

echo "✓ Conda environment activated"
echo "  Python: $(which python)"
echo "  Snakemake: $(which snakemake)"

# Change to repo directory
cd "$REPO_ROOT"
echo ""
echo "Working directory: $(pwd)"

# Create temporary config with batch_id
TEMP_CONFIG=$(mktemp /tmp/snakemake_config_${BATCH_ID}_XXXXXX.yaml)
cp "$CONFIG_FILE" "$TEMP_CONFIG"

# Update config with batch_id and mode
sed -i "s/^mode:.*/mode: single/" "$TEMP_CONFIG"
sed -i "s/^batch_id:.*/batch_id: \"$BATCH_ID\"/" "$TEMP_CONFIG"

echo "✓ Temporary config created: $TEMP_CONFIG"

# Display Snakemake configuration
echo ""
echo "Snakemake Configuration:"
echo "  Snakefile:        $SNAKEFILE"
echo "  Config:           $TEMP_CONFIG"
echo "  Max parallel:     12 jobs"
echo "  Resources/job:    4 cores, 16GB RAM"

# Validate batch exists on Ceres
CERES_BATCH_PATH="/90daydata/dash_agir/data/semifield-upload/$BATCH_ID"
if [ ! -d "$CERES_BATCH_PATH" ]; then
    echo ""
    echo "❌ Error: Batch not found on Ceres: $CERES_BATCH_PATH"
    echo ""
    echo "Pipeline Status:"
    sqlite3 "$DB_PATH" <<EOF
SELECT 
    'Batch: ' || batch_id,
    'NCSU sync: ' || COALESCE(ncsu_sync_status, 'N/A'),
    'Transfer: ' || COALESCE(transfer_status, 'N/A'),
    'Processing: ' || COALESCE(processing_status, 'N/A')
FROM batches 
WHERE batch_id = '$BATCH_ID';
EOF
    
    echo ""
    echo "Did you run the transfer step?"
    echo "  ./scripts/workflow.sh transfer $BATCH_ID"
    exit 1
fi

RAW_COUNT=$(find "$CERES_BATCH_PATH" -name "*.ARW" | wc -l)
echo "✓ Batch found on Ceres: $RAW_COUNT RAW files"

# Check database transfer status
if [ -f "$DB_PATH" ]; then
    TRANSFER_STATUS=$(sqlite3 "$DB_PATH" "SELECT transfer_status FROM batches WHERE batch_id = '$BATCH_ID'" | head -1)
    if [ "$TRANSFER_STATUS" != "transferred" ]; then
        echo "⚠️  Warning: Database shows transfer_status='$TRANSFER_STATUS' (expected 'transferred')"
        echo "   Continuing anyway since files are present on Ceres"
    else
        echo "✓ Database confirms transfer complete"
    fi
fi

# Run Snakemake
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Starting Snakemake Pipeline"
echo "═══════════════════════════════════════════════════════════════"
echo ""

snakemake \
    --snakefile "$SNAKEFILE" \
    --configfile "$TEMP_CONFIG" \
    --cores all \
    --jobs 12 \
    --executor slurm \
    --default-resources slurm_account=dash_agir \
    --default-resources slurm_partition=short \
    --default-resources mem_mb=16000 \
    --default-resources runtime=120 \
    --latency-wait 60 \
    --rerun-incomplete \
    --keep-going \
    --printshellcmds

SNAKEMAKE_EXIT=$?

# Cleanup temporary config
rm -f "$TEMP_CONFIG"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Pipeline Finished"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Exit code: $SNAKEMAKE_EXIT"
echo "Completed: $(date)"

# Update database based on exit code
if [ -f "$DB_PATH" ]; then
    if [ $SNAKEMAKE_EXIT -eq 0 ]; then
        sqlite3 "$DB_PATH" <<EOF
UPDATE batches 
SET processing_status = 'completed',
    processing_completed_at = datetime('now'),
    updated_at = datetime('now')
WHERE batch_id = '$BATCH_ID';

UPDATE processing_history 
SET status = 'completed',
    completed_at = datetime('now')
WHERE batch_id = '$BATCH_ID' 
AND job_id = '$SLURM_JOB_ID';
EOF
        echo "✅ Database updated: processing_status = 'completed'"
    else
        sqlite3 "$DB_PATH" <<EOF
UPDATE batches 
SET processing_status = 'failed',
    updated_at = datetime('now')
WHERE batch_id = '$BATCH_ID';

UPDATE processing_history 
SET status = 'failed',
    completed_at = datetime('now'),
    error_message = 'Snakemake exited with code $SNAKEMAKE_EXIT'
WHERE batch_id = '$BATCH_ID' 
AND job_id = '$SLURM_JOB_ID';
EOF
        echo "❌ Database updated: processing_status = 'failed'"
    fi
fi

# Print summary
echo ""
echo "Summary:"
echo "  Batch ID:         $BATCH_ID"
echo "  SLURM Job:        $SLURM_JOB_ID"
echo "  Exit code:        $SNAKEMAKE_EXIT"

if [ -f "$DB_PATH" ]; then
    echo ""
    echo "Database Status:"
    sqlite3 -box "$DB_PATH" <<EOF
SELECT 
    batch_id,
    ncsu_sync_status,
    transfer_status,
    processing_status,
    processing_completed_at
FROM batches 
WHERE batch_id = '$BATCH_ID';
EOF
fi

# Check output
OUTPUT_DIR="/project/dash_agir/matthew.kutugata/semifield-developed-images/$BATCH_ID"
if [ -d "$OUTPUT_DIR" ]; then
    JPG_COUNT=$(find "$OUTPUT_DIR" -name "*.jpg" | wc -l)
    echo ""
    echo "Output:"
    echo "  Directory:        $OUTPUT_DIR"
    echo "  JPG files:        $JPG_COUNT"
    echo "  Success rate:     $(echo "scale=1; $JPG_COUNT * 100 / $RAW_COUNT" | bc)%"
fi

# Exit with Snakemake's exit code
exit $SNAKEMAKE_EXIT
