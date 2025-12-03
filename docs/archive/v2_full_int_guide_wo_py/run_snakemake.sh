#!/bin/bash
#SBATCH --job-name=svs_raw_process
#SBATCH --partition=short
#SBATCH --account=dash_agir
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --output=/project/dash_agir/matthew.kutugata/logs/snakemake_%x_%j.out
#SBATCH --error=/project/dash_agir/matthew.kutugata/logs/snakemake_%x_%j.err

#
# SLURM Script for SVS RAW Image Processing with Snakemake
# Runs on SCINet Ceres compute nodes
#
# Usage:
#   sbatch -A dash_agir run_snakemake.sh MD_2025-10-22
#   sbatch -A dash_agir run_snakemake.sh MD_2025-10-22 /path/to/custom/config.yaml
#

set -e
set -u
set -o pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}     SVS RAW Processing Pipeline - Snakemake on Ceres        ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get batch ID from command line
BATCH_ID="${1:-}"
if [ -z "$BATCH_ID" ]; then
    echo -e "${RED}❌ Error: Batch ID required${NC}"
    echo "Usage: sbatch run_snakemake.sh <BATCH_ID> [config.yaml]"
    exit 1
fi

# Configuration file (optional second argument)
CONFIG_FILE="${2:-config/snakemake_config.yaml}"

# Repository root
REPO_ROOT="/home/matthew.kutugata/repos/svs-raw-api"
cd "$REPO_ROOT"

# Conda environment
CONDA_ENV="/project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep"

# Database
DB_PATH="/project/dash_agir/matthew.kutugata/pipeline_tracking.db"

# Log file
LOG_DIR="/project/dash_agir/matthew.kutugata/logs"
mkdir -p "$LOG_DIR"

echo -e "${GREEN}Configuration:${NC}"
echo "  Batch ID:    $BATCH_ID"
echo "  Config file: $CONFIG_FILE"
echo "  SLURM Job:   $SLURM_JOB_ID"
echo "  Node:        $(hostname)"
echo "  Cores:       $SLURM_CPUS_PER_TASK"
echo "  Memory:      $SLURM_MEM_PER_NODE MB"
echo ""

# Activate conda environment
echo -e "${BLUE}Activating conda environment...${NC}"
source "$CONDA_ENV/bin/activate"

# Verify key tools
echo -e "${BLUE}Verifying tools...${NC}"
python --version
snakemake --version

# Check if RawTherapee CLI exists
RT_CLI="/home/matthew.kutugata/SemiF-Preprocessing/squashfs-root/usr/bin/rawtherapee-cli"
if [ ! -f "$RT_CLI" ]; then
    echo -e "${RED}❌ RawTherapee CLI not found: $RT_CLI${NC}"
    exit 1
fi
echo -e "${GREEN}✅ RawTherapee CLI found${NC}"

# Check if svs_raw_api is available
if ! python -c "import svs_raw_api" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Installing svs_raw_api...${NC}"
    cd "$REPO_ROOT"
    pip install -e . --no-deps
fi
echo -e "${GREEN}✅ svs_raw_api available${NC}"

# Check if batch exists in scratch
SCRATCH_DIR="/90daydata/dash_agir/data/semifield-upload/$BATCH_ID"
if [ ! -d "$SCRATCH_DIR" ]; then
    echo -e "${RED}❌ Batch directory not found: $SCRATCH_DIR${NC}"
    echo "   Make sure to transfer from JUNO first!"
    exit 1
fi

# Count RAW files
RAW_COUNT=$(find "$SCRATCH_DIR" -name "*.ARW" -o -name "*.RAW" | wc -l)
echo -e "${GREEN}✅ Found $RAW_COUNT RAW files in $BATCH_ID${NC}"
echo ""

# Update database: mark as processing
echo -e "${BLUE}Updating database status...${NC}"
if [ -f "$REPO_ROOT/scripts/db_manager.py" ]; then
    python "$REPO_ROOT/scripts/db_manager.py" --db "$DB_PATH" update-status \
        --batch-id "$BATCH_ID" \
        --field processing_status \
        --value processing || true
fi

# Create output directories
OUTPUT_BASE="/project/dash_agir/matthew.kutugata/semifield-developed-images/$BATCH_ID"
mkdir -p "$OUTPUT_BASE/dngs"
mkdir -p "$OUTPUT_BASE/images"
mkdir -p "$OUTPUT_BASE/logs"

echo -e "${BLUE}Starting Snakemake workflow...${NC}"
echo ""

# Run Snakemake
# - Cluster mode with SLURM
# - Up to 12 parallel jobs
# - Each job gets 4 cores and 16GB RAM
SNAKEMAKE_LOG="$LOG_DIR/snakemake_${BATCH_ID}_${SLURM_JOB_ID}.log"

snakemake \
    --snakefile Snakefile \
    --configfile "$CONFIG_FILE" \
    --config batch_id="$BATCH_ID" mode=single \
    --cores $SLURM_CPUS_PER_TASK \
    --jobs 12 \
    --latency-wait 60 \
    --restart-times 2 \
    --keep-going \
    --printshellcmds \
    --reason \
    --use-conda \
    2>&1 | tee "$SNAKEMAKE_LOG"

SNAKEMAKE_EXIT=${PIPESTATUS[0]}

echo ""
if [ $SNAKEMAKE_EXIT -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}                   Processing Complete!                       ${GREEN}║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    
    # Count output files
    JPG_COUNT=$(find "$OUTPUT_BASE/images" -name "*.jpg" 2>/dev/null | wc -l)
    DNG_COUNT=$(find "$OUTPUT_BASE/dngs" -name "*.dng" 2>/dev/null | wc -l)
    
    echo ""
    echo "Results:"
    echo "  RAW files:  $RAW_COUNT"
    echo "  DNG files:  $DNG_COUNT"
    echo "  JPG files:  $JPG_COUNT"
    echo "  Output:     $OUTPUT_BASE/images/"
    echo ""
    
    # Update database: mark as completed
    if [ -f "$REPO_ROOT/scripts/db_manager.py" ]; then
        python "$REPO_ROOT/scripts/db_manager.py" --db "$DB_PATH" update-status \
            --batch-id "$BATCH_ID" \
            --field processing_status \
            --value completed || true
        
        python "$REPO_ROOT/scripts/db_manager.py" --db "$DB_PATH" update-status \
            --batch-id "$BATCH_ID" \
            --field processing_completed_at \
            --value "$(date -Iseconds)" || true
    fi
    
    # Copy final images to JUNO archive (optional)
    JUNO_OUTPUT="/project/dash_agir/semifield-upload/$BATCH_ID/developed"
    if [ ! -d "$JUNO_OUTPUT" ]; then
        echo -e "${BLUE}Archiving JPGs to JUNO...${NC}"
        mkdir -p "$JUNO_OUTPUT"
        rsync -av "$OUTPUT_BASE/images/" "$JUNO_OUTPUT/" || echo -e "${YELLOW}⚠️  Could not copy to JUNO${NC}"
    fi
    
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║${NC}                   Processing Failed!                         ${RED}║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Check logs:"
    echo "  $SNAKEMAKE_LOG"
    echo "  $OUTPUT_BASE/logs/"
    echo ""
    
    # Update database: mark as failed
    if [ -f "$REPO_ROOT/scripts/db_manager.py" ]; then
        python "$REPO_ROOT/scripts/db_manager.py" --db "$DB_PATH" update-status \
            --batch-id "$BATCH_ID" \
            --field processing_status \
            --value failed || true
    fi
    
    exit 1
fi
