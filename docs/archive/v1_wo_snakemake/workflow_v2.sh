#!/bin/bash
#
# Enhanced Three-Tier Pipeline Workflow Manager
# NCSU → JUNO → Ceres → Process
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

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    local errors=0
    
    # Check Globus CLI
    if ! command -v globus &> /dev/null; then
        print_error "Globus CLI not found. Install: pip install globus-cli"
        errors=$((errors + 1))
    else
        # Check login status
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
    
    return $errors
}

# Initialize database
init_database() {
    print_info "Initializing database..."
    source "$CONDA_ENV/bin/activate"
    python "$DB_MANAGER" --db "$DB_PATH" init
    print_success "Database initialized"
}

# Check for missing batches (NCSU vs JUNO)
check_missing_batches() {
    print_header "CHECKING FOR BATCHES NEEDING SYNC (NCSU → JUNO)"
    
    source "$CONDA_ENV/bin/activate"
    
    local state_filter=""
    if [ -n "$1" ]; then
        state_filter="--state $1"
    fi
    
    python "$GLOBUS_MANAGER" check-missing $state_filter --db "$DB_PATH"
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

# Sync all missing batches from NCSU to JUNO
sync_all_to_juno() {
    print_header "SYNCING ALL MISSING BATCHES: NCSU → JUNO"
    
    source "$CONDA_ENV/bin/activate"
    
    local state_filter=""
    if [ -n "$1" ]; then
        state_filter="--state $1"
    fi
    
    python "$GLOBUS_MANAGER" full-sync $state_filter --db "$DB_PATH"
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

# Process single batch (assumes already on Ceres)
process_batch() {
    local batch_id="$1"
    
    if [ -z "$batch_id" ]; then
        print_error "Batch ID required"
        echo "Usage: $0 process <BATCH_ID>"
        exit 1
    fi
    
    print_header "PROCESSING BATCH: $batch_id"
    
    # Update config for single batch mode
    local config_file="$REPO_ROOT/config/snakemake_config_enhanced.yaml"
    
    if [ ! -f "$config_file" ]; then
        print_error "Config file not found: $config_file"
        exit 1
    fi
    
    # Create temporary config with batch_id
    local temp_config="/tmp/snakemake_config_${batch_id}.yaml"
    sed "s/^mode:.*/mode: single/" "$config_file" > "$temp_config"
    sed -i "s/^batch_id:.*/batch_id: $batch_id/" "$temp_config"
    
    # Submit SLURM job
    local slurm_script="$REPO_ROOT/slurm/run_snakemake_enhanced.sh"
    
    if [ ! -f "$slurm_script" ]; then
        print_error "SLURM script not found: $slurm_script"
        exit 1
    fi
    
    print_info "Submitting SLURM job..."
    sbatch -A dash_agir "$slurm_script" "$batch_id" "$temp_config"
    
    if [ $? -eq 0 ]; then
        print_success "Processing job submitted for $batch_id"
    else
        print_error "Failed to submit processing job"
        exit 1
    fi
}

# Full pipeline for single batch (sync → transfer → process)
full_pipeline_single() {
    local batch_id="$1"
    
    if [ -z "$batch_id" ]; then
        print_error "Batch ID required"
        echo "Usage: $0 full-pipeline <BATCH_ID>"
        exit 1
    fi
    
    print_header "FULL PIPELINE FOR BATCH: $batch_id"
    
    # Step 1: Check if needs NCSU sync
    print_info "Step 1/3: Checking NCSU sync status..."
    source "$CONDA_ENV/bin/activate"
    
    local needs_sync=$(python "$DB_MANAGER" --db "$DB_PATH" list | grep "$batch_id" | grep -c "needed\|unknown" || true)
    
    if [ "$needs_sync" -gt 0 ]; then
        print_info "Batch needs sync from NCSU → JUNO"
        sync_batch_to_juno "$batch_id"
        
        print_info "Waiting for sync to complete..."
        sleep 60  # Give it a minute before checking
        
        # TODO: Add proper wait loop checking Globus task status
    else
        print_info "Batch already in JUNO, skipping sync"
    fi
    
    # Step 2: Transfer JUNO → Ceres
    print_info "Step 2/3: Transferring JUNO → Ceres..."
    transfer_batch_to_ceres "$batch_id"
    
    print_info "Waiting for transfer to complete..."
    sleep 60
    
    # Step 3: Process on Ceres
    print_info "Step 3/3: Processing on Ceres..."
    process_batch "$batch_id"
    
    print_success "Full pipeline initiated for $batch_id"
}

# Show pipeline status
show_status() {
    print_header "PIPELINE STATUS"
    
    source "$CONDA_ENV/bin/activate"
    python "$DB_MANAGER" --db "$DB_PATH" summary
    
    echo ""
    print_info "Batches needing NCSU sync:"
    python "$DB_MANAGER" --db "$DB_PATH" list | grep -E "needed|unknown" || print_info "None"
    
    echo ""
    print_info "Batches ready for transfer:"
    python "$DB_MANAGER" --db "$DB_PATH" list | grep "pending.*pending" || print_info "None"
    
    echo ""
    print_info "Recent SLURM jobs:"
    squeue -u $USER -A dash_agir --format="%.10i %.9P %.30j %.8T %.10M %.6D" | head -20
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

# Interactive menu
interactive_menu() {
    while true; do
        clear
        print_header "SVS RAW PROCESSING PIPELINE - THREE-TIER SYSTEM"
        echo ""
        echo "Storage Tiers:"
        echo "  1. NCSU NFS (Primary upload location)"
        echo "  2. JUNO Archive (Long-term storage)"
        echo "  3. Ceres /90daydata (Processing scratch space)"
        echo ""
        echo "Pipeline Stages:"
        echo "  NCSU → JUNO → Ceres → Process → Archive"
        echo ""
        echo "Commands:"
        echo "  1) Check missing batches (NCSU vs JUNO)"
        echo "  2) Sync single batch (NCSU → JUNO)"
        echo "  3) Sync all missing batches"
        echo "  4) Transfer batch (JUNO → Ceres)"
        echo "  5) Process batch on Ceres"
        echo "  6) Full pipeline for single batch"
        echo "  7) Show pipeline status"
        echo "  8) Check Globus task"
        echo "  9) Exit"
        echo ""
        read -p "Select option: " choice
        
        case $choice in
            1)
                read -p "State filter (optional, e.g., MD): " state
                check_missing_batches "$state"
                read -p "Press enter to continue..."
                ;;
            2)
                read -p "Batch ID (e.g., MD_2025-10-22): " batch_id
                sync_batch_to_juno "$batch_id"
                read -p "Press enter to continue..."
                ;;
            3)
                read -p "State filter (optional): " state
                sync_all_to_juno "$state"
                read -p "Press enter to continue..."
                ;;
            4)
                read -p "Batch ID: " batch_id
                transfer_batch_to_ceres "$batch_id"
                read -p "Press enter to continue..."
                ;;
            5)
                read -p "Batch ID: " batch_id
                process_batch "$batch_id"
                read -p "Press enter to continue..."
                ;;
            6)
                read -p "Batch ID: " batch_id
                full_pipeline_single "$batch_id"
                read -p "Press enter to continue..."
                ;;
            7)
                show_status
                read -p "Press enter to continue..."
                ;;
            8)
                read -p "Globus task ID: " task_id
                check_task "$task_id"
                read -p "Press enter to continue..."
                ;;
            9)
                print_success "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid option"
                sleep 2
                ;;
        esac
    done
}

# Show usage
show_usage() {
    cat << EOF
Enhanced Three-Tier Pipeline Workflow Manager
NCSU → JUNO → Ceres → Process

Usage: $0 <command> [options]

Commands:
  check-missing [STATE]          Check for batches in NCSU but not JUNO
  sync <BATCH_ID>                Sync single batch from NCSU to JUNO
  sync-all [STATE]               Sync all missing batches to JUNO
  transfer <BATCH_ID>            Transfer batch from JUNO to Ceres
  process <BATCH_ID>             Process batch on Ceres
  full-pipeline <BATCH_ID>       Run complete pipeline for one batch
  status                         Show overall pipeline status
  check-task <TASK_ID>           Check Globus task status
  interactive                    Launch interactive menu
  init                          Initialize database
  help                          Show this help

Examples:
  # Check what needs syncing
  $0 check-missing
  $0 check-missing MD

  # Sync specific batch from NCSU to JUNO
  $0 sync MD_2025-10-22

  # Transfer from JUNO to Ceres
  $0 transfer MD_2025-10-22

  # Process on Ceres
  $0 process MD_2025-10-22

  # Run complete pipeline
  $0 full-pipeline MD_2025-10-22

  # Check status
  $0 status

  # Check Globus transfer
  $0 check-task <task-id>

  # Interactive mode
  $0 interactive

Storage Locations:
  NCSU:  [Configure in globus_manager.py]
  JUNO:  /project/dash_agir/semifield-upload
  Ceres: /90daydata/dash_agir/data/semifield-upload

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
        sync-all)
            sync_all_to_juno "$@"
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
        interactive)
            interactive_menu
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
