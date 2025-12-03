#!/bin/bash
#
# Enhanced Three-Tier Pipeline Workflow Manager with Image Processing
# NCSU → JUNO → Ceres → Process → Archive
#
# Usage: ./workflow.sh [command] [options]
#

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DB_PATH="/project/dash_agir/matthew.kutugata/pipeline_tracking.db"
CONDA_ENV="/project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep"
GLOBUS_MANAGER="$REPO_ROOT/scripts/globus_manager.py"
DB_MANAGER="$REPO_ROOT/scripts/db_manager.py"
SNAKEMAKE_SCRIPT="$REPO_ROOT/slurm/run_snakemake.sh"
CONFIG_FILE="$REPO_ROOT/config/snakemake_config.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Print functions
print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    local errors=0
    
    # Check Globus CLI
    if ! command -v globus &> /dev/null; then
        print_error "Globus CLI not found. Install: pip install globus-cli"
        errors=$((errors + 1))
    else
        if ! globus whoami &> /dev/null; then
            print_error "Not logged into Globus. Run: globus login"
            errors=$((errors + 1))
        fi
    fi
    
    # Check conda environment
    if [ ! -d "$CONDA_ENV" ]; then
        print_error "Conda environment not found: $CONDA_ENV"
        errors=$((errors + 1))
    fi
    
    # Check scripts exist
    if [ ! -f "$GLOBUS_MANAGER" ]; then
        print_error "Globus manager not found: $GLOBUS_MANAGER"
        errors=$((errors + 1))
    fi
    
    if [ ! -f "$DB_MANAGER" ]; then
        print_error "Database manager not found: $DB_MANAGER"
        errors=$((errors + 1))
    fi
    
    # Check SLURM script
    if [ ! -f "$SNAKEMAKE_SCRIPT" ]; then
        print_warning "SLURM script not found: $SNAKEMAKE_SCRIPT"
        print_info "Processing commands will not work"
    fi
    
    # Check Snakemake config
    if [ ! -f "$CONFIG_FILE" ]; then
        print_warning "Snakemake config not found: $CONFIG_FILE"
    fi
    
    return $errors
}

# Process single batch
process_batch() {
    local batch_id="$1"
    
    if [ -z "$batch_id" ]; then
        print_error "Batch ID required"
        echo "Usage: $0 process <BATCH_ID>"
        exit 1
    fi
    
    print_header "PROCESSING BATCH: $batch_id"
    
    # Check if SLURM script exists
    if [ ! -f "$SNAKEMAKE_SCRIPT" ]; then
        print_error "SLURM script not found: $SNAKEMAKE_SCRIPT"
        exit 1
    fi
    
    # Check if batch exists on Ceres
    local ceres_path="/90daydata/dash_agir/data/semifield-upload/$batch_id"
    if [ ! -d "$ceres_path" ]; then
        print_error "Batch not found on Ceres: $ceres_path"
        print_info "Transfer batch first: $0 transfer $batch_id"
        exit 1
    fi
    
    # Count RAW files
    local raw_count=$(find "$ceres_path" -name "*.ARW" -o -name "*.RAW" 2>/dev/null | wc -l)
    print_info "Found $raw_count RAW files in $batch_id"
    
    if [ $raw_count -eq 0 ]; then
        print_error "No RAW files found in $ceres_path"
        exit 1
    fi
    
    # Submit SLURM job
    print_info "Submitting SLURM job for image processing..."
    
    local job_output
    job_output=$(sbatch -A dash_agir "$SNAKEMAKE_SCRIPT" "$batch_id" "$CONFIG_FILE" 2>&1)
    
    if [ $? -eq 0 ]; then
        local job_id=$(echo "$job_output" | grep -oP 'Submitted batch job \K\d+')
        print_success "Processing job submitted: Job ID $job_id"
        echo ""
        print_info "Monitor job with:"
        echo "  squeue -j $job_id"
        echo "  sacct -j $job_id --format=JobID,JobName,State,Elapsed"
        echo ""
        print_info "Check logs:"
        echo "  tail -f /project/dash_agir/matthew.kutugata/logs/snakemake_svs_raw_process_${job_id}.out"
    else
        print_error "Failed to submit processing job"
        echo "$job_output"
        exit 1
    fi
}

# Transfer batch from JUNO to Ceres
transfer_batch_to_ceres() {
    local batch_id="$1"
    
    if [ -z "$batch_id" ]; then
        print_error "Batch ID required"
        echo "Usage: $0 transfer <BATCH_ID>"
        exit 1
    fi
    
    print_header "TRANSFERRING BATCH: JUNO → Ceres"
    print_info "Batch: $batch_id"
    
    source "$CONDA_ENV/bin/activate"
    python "$GLOBUS_MANAGER" transfer-to-ceres --batch-id "$batch_id" --db "$DB_PATH"
    
    if [ $? -eq 0 ]; then
        print_success "Transfer submitted for $batch_id"
    else
        print_error "Failed to submit transfer for $batch_id"
        exit 1
    fi
}

# Sync single batch from NCSU to JUNO
sync_batch_to_juno() {
    local batch_id="$1"
    
    if [ -z "$batch_id" ]; then
        print_error "Batch ID required"
        echo "Usage: $0 sync <BATCH_ID>"
        exit 1
    fi
    
    print_header "SYNCING BATCH: NCSU → JUNO"
    print_info "Batch: $batch_id"
    
    source "$CONDA_ENV/bin/activate"
    python "$GLOBUS_MANAGER" sync-to-juno --batch-id "$batch_id" --db "$DB_PATH"
    
    if [ $? -eq 0 ]; then
        print_success "Sync submitted for $batch_id"
    else
        print_error "Failed to submit sync for $batch_id"
        exit 1
    fi
}

# Check for missing batches
check_missing_batches() {
    print_header "CHECKING FOR BATCHES NEEDING SYNC (NCSU → JUNO)"
    
    source "$CONDA_ENV/bin/activate"
    
    local state_filter=""
    if [ -n "$1" ]; then
        state_filter="--state $1"
    fi
    
    python "$GLOBUS_MANAGER" check-missing $state_filter --db "$DB_PATH"
}

# Full pipeline for single batch
full_pipeline_single() {
    local batch_id="$1"
    
    if [ -z "$batch_id" ]; then
        print_error "Batch ID required"
        echo "Usage: $0 full-pipeline <BATCH_ID>"
        exit 1
    fi
    
    print_header "FULL PIPELINE FOR BATCH: $batch_id"
    
    # Step 1: Check if needs NCSU sync
    print_info "Step 1/4: Checking NCSU sync status..."
    source "$CONDA_ENV/bin/activate"
    
    local needs_sync=$(python "$DB_MANAGER" --db "$DB_PATH" list 2>/dev/null | grep "$batch_id" | grep -c "needed\|unknown" || true)
    
    if [ "$needs_sync" -gt 0 ]; then
        print_info "Batch needs sync from NCSU → JUNO"
        sync_batch_to_juno "$batch_id"
        
        print_info "Waiting for sync to complete (this may take 10-30 minutes)..."
        sleep 60
        # TODO: Add proper wait loop checking Globus task status
    else
        print_success "Batch already in JUNO, skipping sync"
    fi
    
    # Step 2: Transfer JUNO → Ceres
    print_info "Step 2/4: Transferring JUNO → Ceres..."
    transfer_batch_to_ceres "$batch_id"
    
    print_info "Waiting for transfer to complete (this may take 5-10 minutes)..."
    sleep 60
    
    # Step 3: Verify transfer complete (check if directory exists)
    local ceres_path="/90daydata/dash_agir/data/semifield-upload/$batch_id"
    local wait_count=0
    while [ ! -d "$ceres_path" ] && [ $wait_count -lt 20 ]; do
        print_info "Waiting for transfer... ($wait_count/20)"
        sleep 30
        wait_count=$((wait_count + 1))
    done
    
    if [ ! -d "$ceres_path" ]; then
        print_error "Transfer did not complete in time"
        print_info "Check transfer status with: globus task list"
        exit 1
    fi
    
    print_success "Batch transferred to Ceres"
    
    # Step 4: Process on Ceres
    print_info "Step 4/4: Processing on Ceres..."
    process_batch "$batch_id"
    
    print_success "Full pipeline initiated for $batch_id"
    echo ""
    print_info "Processing will complete in background (SLURM job)"
    print_info "Monitor with: squeue -u \$USER -A dash_agir"
}

# Show pipeline status
show_status() {
    print_header "PIPELINE STATUS"
    
    source "$CONDA_ENV/bin/activate"
    
    # Database summary
    if [ -f "$DB_MANAGER" ]; then
        python "$DB_MANAGER" --db "$DB_PATH" summary 2>/dev/null || print_warning "Could not get database summary"
    fi
    
    echo ""
    print_info "Recent SLURM jobs:"
    squeue -u $USER -A dash_agir --format="%.10i %.9P %.30j %.8T %.10M %.6D" 2>/dev/null | head -20 || print_warning "Could not get SLURM jobs"
    
    echo ""
    print_info "Recent completed jobs (last 24h):"
    sacct -X -u $USER -A dash_agir --starttime=now-1day --format=JobID,JobName,State,Elapsed 2>/dev/null | head -10 || print_warning "Could not get job history"
}

# Check Globus task status
check_task() {
    local task_id="$1"
    
    if [ -z "$task_id" ]; then
        print_error "Task ID required"
        echo "Usage: $0 check-task <TASK_ID>"
        exit 1
    fi
    
    print_header "GLOBUS TASK STATUS: $task_id"
    
    source "$CONDA_ENV/bin/activate"
    python "$GLOBUS_MANAGER" status --task-id "$task_id" --db "$DB_PATH"
}

# Initialize database
init_database() {
    print_info "Initializing database..."
    source "$CONDA_ENV/bin/activate"
    python "$DB_MANAGER" --db "$DB_PATH" init
    print_success "Database initialized"
}

# Show usage
show_usage() {
    cat << EOF
Enhanced Three-Tier Pipeline Workflow Manager
NCSU → JUNO → Ceres → Process → Archive

Usage: $0 <command> [options]

Commands:
  check-missing [STATE]          Check for batches in NCSU but not JUNO
  sync <BATCH_ID>                Sync single batch from NCSU to JUNO
  transfer <BATCH_ID>            Transfer batch from JUNO to Ceres
  process <BATCH_ID>             Process batch on Ceres (RAW → DNG → JPG)
  full-pipeline <BATCH_ID>       Run complete pipeline for one batch
  status                         Show overall pipeline status
  check-task <TASK_ID>           Check Globus task status
  init                          Initialize database
  help                          Show this help

Examples:
  # Check what needs syncing from NCSU
  $0 check-missing
  $0 check-missing MD

  # Sync specific batch from NCSU to JUNO
  $0 sync MD_2025-10-22

  # Transfer from JUNO to Ceres
  $0 transfer MD_2025-10-22

  # Process images on Ceres (submits SLURM job)
  $0 process MD_2025-10-22

  # Run complete pipeline
  $0 full-pipeline MD_2025-10-22

  # Check status
  $0 status

  # Check Globus transfer
  $0 check-task <task-id>

Storage Locations:
  NCSU:   [Configure in globus_manager.py]
  JUNO:   /project/dash_agir/semifield-upload
  Ceres:  /90daydata/dash_agir/data/semifield-upload
  Output: /project/dash_agir/matthew.kutugata/semifield-developed-images

Processing:
  - RAW files → DNG (Adobe format)
  - DNG files → JPG (RawTherapee)
  - Parallel processing: up to 12 simultaneous images
  - Resources: 4 cores × 16GB RAM per image

Database: $DB_PATH
EOF
}

# Main command router
main() {
    # Check prerequisites first
    if ! check_prerequisites; then
        print_error "Prerequisites check failed"
        exit 1
    fi
    
    # Parse command
    local command="${1:-help}"
    shift || true
    
    case "$command" in
        check-missing)
            check_missing_batches "$@"
            ;;
        sync)
            sync_batch_to_juno "$@"
            ;;
        transfer)
            transfer_batch_to_ceres "$@"
            ;;
        process)
            process_batch "$@"
            ;;
        full-pipeline)
            full_pipeline_single "$@"
            ;;
        status)
            show_status
            ;;
        check-task)
            check_task "$@"
            ;;
        init)
            init_database
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
