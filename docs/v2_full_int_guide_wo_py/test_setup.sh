#!/bin/bash
#
# Pipeline Setup Verification Script
# Tests that all components are properly configured
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}     SVS RAW Processing Pipeline - Setup Verification       ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

ERRORS=0
WARNINGS=0

# Test function
test_item() {
    local test_name="$1"
    local test_command="$2"
    local required="${3:-true}"
    
    echo -n "Testing: $test_name ... "
    
    if eval "$test_command" &>/dev/null; then
        echo -e "${GREEN}✅${NC}"
        return 0
    else
        if [ "$required" = "true" ]; then
            echo -e "${RED}❌ FAILED${NC}"
            ERRORS=$((ERRORS + 1))
            return 1
        else
            echo -e "${YELLOW}⚠️  WARNING${NC}"
            WARNINGS=$((WARNINGS + 1))
            return 1
        fi
    fi
}

# Section: Prerequisites
echo -e "${BLUE}=== Prerequisites ===${NC}"

test_item "Globus CLI installed" "command -v globus"
test_item "Globus logged in" "globus whoami"
test_item "Snakemake installed" "snakemake --version"
test_item "Python 3 available" "python3 --version"

echo ""

# Section: Conda Environment
echo -e "${BLUE}=== Conda Environment ===${NC}"

CONDA_ENV="/project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep"
test_item "Conda environment exists" "[ -d $CONDA_ENV ]"

# Activate and test packages
if [ -d "$CONDA_ENV" ]; then
    source "$CONDA_ENV/bin/activate" 2>/dev/null
    test_item "numpy installed" "python -c 'import numpy'"
    test_item "svs_raw_api installed" "python -c 'import svs_raw_api'" false
fi

echo ""

# Section: File Structure
echo -e "${BLUE}=== File Structure ===${NC}"

REPO_ROOT="/home/$USER/repos/svs-raw-api"
test_item "Repository root exists" "[ -d $REPO_ROOT ]"
test_item "Snakefile exists" "[ -f $REPO_ROOT/Snakefile ]"
test_item "Snakemake config exists" "[ -f $REPO_ROOT/config/snakemake_config.yaml ]"
test_item "SLURM script exists" "[ -f $REPO_ROOT/slurm/run_snakemake.sh ]"
test_item "SLURM script executable" "[ -x $REPO_ROOT/slurm/run_snakemake.sh ]"
test_item "Workflow script exists" "[ -f $REPO_ROOT/scripts/workflow.sh ]"
test_item "Workflow script executable" "[ -x $REPO_ROOT/scripts/workflow.sh ]"
test_item "Globus manager exists" "[ -f $REPO_ROOT/scripts/globus_manager.py ]"
test_item "Database manager exists" "[ -f $REPO_ROOT/scripts/db_manager.py ]"

echo ""

# Section: Configuration Files
echo -e "${BLUE}=== Configuration Files ===${NC}"

PROFILE_DIR="$REPO_ROOT/data/profiles"
test_item "Profile directory exists" "[ -d $PROFILE_DIR ]"
test_item "SVS tags YAML exists" "[ -f $PROFILE_DIR/svs_tags.yaml ]" false
test_item "Color matrix exists" "[ -f $PROFILE_DIR/MD_calibration_matrix_optimized.npy ]" false
test_item "PP3 profile exists" "[ -f $PROFILE_DIR/MD_shr661_raw16.pp3 ]" false

echo ""

# Section: RawTherapee
echo -e "${BLUE}=== RawTherapee ===${NC}"

RT_CLI="/home/$USER/SemiF-Preprocessing/squashfs-root/usr/bin/rawtherapee-cli"
test_item "RawTherapee CLI exists" "[ -f $RT_CLI ]"
test_item "RawTherapee CLI executable" "[ -x $RT_CLI ]"

echo ""

# Section: Storage Directories
echo -e "${BLUE}=== Storage Directories ===${NC}"

test_item "JUNO archive accessible" "[ -d /project/dash_agir/semifield-upload ]"
test_item "Ceres scratch accessible" "[ -d /90daydata/dash_agir/data ]"
test_item "Output directory exists" "[ -d /project/dash_agir/matthew.kutugata/semifield-developed-images ]"
test_item "Logs directory exists" "[ -d /project/dash_agir/matthew.kutugata/logs ]"

echo ""

# Section: Database
echo -e "${BLUE}=== Database ===${NC}"

DB_PATH="/project/dash_agir/matthew.kutugata/pipeline_tracking.db"
test_item "Database file exists" "[ -f $DB_PATH ]" false

if [ -f "$DB_PATH" ]; then
    test_item "Database readable" "sqlite3 $DB_PATH 'SELECT COUNT(*) FROM batches' > /dev/null" false
fi

echo ""

# Section: SLURM Access
echo -e "${BLUE}=== SLURM Access ===${NC}"

test_item "SLURM commands available" "command -v sbatch"
test_item "Can query SLURM account" "sacctmgr show user $USER -s | grep -q dash_agir" false

echo ""

# Section: Globus Endpoints
echo -e "${BLUE}=== Globus Endpoints ===${NC}"

JUNO_EP="904c2108-90cf-11e8-9672-0a6d4e044368"
CERES_EP="f45a24f8-09ba-11ec-b342-1feaf93e3729"

test_item "Can access JUNO endpoint" "globus ls $JUNO_EP:/project/dash_agir" false
test_item "Can access Ceres endpoint" "globus ls $CERES_EP:/" false

echo ""

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}                     Test Summary                            ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ All critical tests passed!${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠️  $WARNINGS warnings (non-critical issues)${NC}"
        echo ""
        echo "You can proceed with testing, but address warnings for full functionality."
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Test workflow: cd ~/repos/svs-raw-api && ./scripts/workflow.sh status"
    echo "  2. Process test batch: ./scripts/workflow.sh process <BATCH_ID>"
    echo ""
    exit 0
else
    echo -e "${RED}❌ $ERRORS critical tests failed${NC}"
    echo -e "${YELLOW}⚠️  $WARNINGS warnings${NC}"
    echo ""
    echo "Please fix the failed tests before proceeding."
    echo "See PROCESSING_INTEGRATION_GUIDE.md for setup instructions."
    echo ""
    exit 1
fi
