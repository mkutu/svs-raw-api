#!/bin/bash
#SBATCH --job-name=svs_pipeline
#SBATCH --partition=short
#SBATCH --account=dash_agir
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --output=/project/dash_agir/matthew.kutugata/logs/snakemake_%x_%j.out.log
#SBATCH --error=/project/dash_agir/matthew.kutugata/logs/snakemake_%x_%j.err.log

#
# SLURM Script for SVS RAW Image Processing with Snakemake
# Runs on SCINet Ceres compute nodes
#
# Usage:
#   sbatch slurm/run_snakemake.sh <BATCH_ID>
#   sbatch slurm/run_snakemake.sh MD_2025-10-22
#

set -euo pipefail

# Color output for logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}     SVS RAW Processing Pipeline - Snakemake on Ceres        ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# Configuration
# ============================================================================

# Get batch ID from command line
BATCH_ID="${1:-}"
if [ -z "$BATCH_ID" ]; then
    echo -e "${RED}❌ Error: Batch ID required${NC}"
    echo "Usage: sbatch slurm/run_snakemake.sh <BATCH_ID>"
    exit 1
fi

# Paths
REPO_ROOT="/home/matthew.kutugata/repos/svs-raw-api"
CONDA_ENV="/project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep"
CONFIG_FILE="$REPO_ROOT/config/scinet.yaml"
LOG_DIR="/project/dash_agir/matthew.kutugata/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# ============================================================================
# Environment Setup
# ============================================================================

echo -e "${GREEN}Configuration:${NC}"
echo "  Batch ID:    $BATCH_ID"
echo "  Config file: $CONFIG_FILE"
echo "  SLURM Job:   $SLURM_JOB_ID"
echo "  Node:        $(hostname)"
echo "  Cores:       $SLURM_CPUS_PER_TASK"
echo "  Memory:      $SLURM_MEM_PER_NODE MB"
echo "  Start time:  $(date)"
echo ""

# Change to repository directory
cd "$REPO_ROOT"

# Validate RawTherapee path
echo -e "${BLUE}Validating RawTherapee...${NC}"
bash scripts/validate_rawtherapee.sh > scripts/rawtherapee_path.sh
source scripts/rawtherapee_path.sh

if [ ! -f "$RT_CLI_PATH" ]; then
    echo -e "${RED}❌ RawTherapee CLI not found: $RT_CLI_PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✅ RawTherapee CLI: $RT_CLI_PATH${NC}"

# Activate conda environment
echo -e "${BLUE}Activating conda environment...${NC}"
module load miniconda
source activate "$CONDA_ENV"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to activate conda environment${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Conda environment activated${NC}"
python --version
echo ""

# Verify svs_raw_api package
echo -e "${BLUE}Verifying svs_raw_api package...${NC}"
if ! python -c "import svs_raw_api" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Installing svs_raw_api...${NC}"
    pip install -e . --no-deps
fi
echo -e "${GREEN}✅ svs_raw_api available${NC}"
echo ""

# Verify snakemake
echo -e "${BLUE}Verifying Snakemake...${NC}"
snakemake --version
echo ""

# ============================================================================
# Validate Input
# ============================================================================

echo -e "${BLUE}Validating batch...${NC}"

SCRATCH_DIR="/90daydata/dash_agir/data/semifield-upload/$BATCH_ID"
if [ ! -d "$SCRATCH_DIR" ]; then
    echo -e "${RED}❌ Batch directory not found: $SCRATCH_DIR${NC}"
    echo "   Make sure to transfer from JUNO first!"
    exit 1
fi

# Count RAW files
RAW_COUNT=$(find "$SCRATCH_DIR" -maxdepth 1 \( -name "*.ARW" -o -name "*.RAW" \) | wc -l)
if [ "$RAW_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ No RAW files found in $SCRATCH_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found $RAW_COUNT RAW files in batch $BATCH_ID${NC}"
echo ""

# ============================================================================
# Run Snakemake
# ============================================================================

echo -e "${BLUE}Starting Snakemake workflow...${NC}"
echo ""

# Temporary directory inside node is already set by SLURM to $TMPDIR
export TMPDIR="$TMPDIR/svs_snakemake_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
echo "Using temporary directory: $TMPDIR"
echo ""

# Snakemake command
snakemake \
    --config batch_id="$BATCH_ID" \
    --configfile "$CONFIG_FILE" \
    --cores "$SLURM_CPUS_PER_TASK" \
    --use-conda \
    --keep-going \
    --rerun-incomplete \
    --printshellcmds 
    # --report "$LOG_DIR/snakemake_stats_${BATCH_ID}_${SLURM_JOB_ID}.html"

EXIT_CODE=$?

# ============================================================================
# Summary
# ============================================================================

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}                    Processing Complete                       ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Snakemake completed successfully${NC}"
    
    # Display summary
    SUMMARY_FILE="/project/dash_agir/matthew.kutugata/semifield-developed-images/$BATCH_ID/processing_summary.txt"
    if [ -f "$SUMMARY_FILE" ]; then
        echo ""
        cat "$SUMMARY_FILE"
    fi
else
    echo -e "${RED}❌ Snakemake failed with exit code $EXIT_CODE${NC}"
    echo ""
    echo "Check logs:"
    echo "  SLURM output: $LOG_DIR/snakemake_svs_pipeline_${SLURM_JOB_ID}.out.log"
    echo "  SLURM errors: $LOG_DIR/snakemake_svs_pipeline_${SLURM_JOB_ID}.err.log"
    echo "  Snakemake log: $LOG_DIR/snakemake_${BATCH_ID}_${SLURM_JOB_ID}.log"
fi

echo ""
echo "End time: $(date)"
echo "Duration: $SECONDS seconds"
echo ""

exit $EXIT_CODE
