#!/bin/bash
#
# Comprehensive Setup Verification Script
# Tests all components of the SVS RAW processing pipeline
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

test_item() {
    local name="$1"
    local command="$2"
    local required="${3:-true}"
    
    echo -n "Testing: $name ... "
    
    if eval "$command" &>/dev/null; then
        echo -e "${GREEN}✅${NC}"
        return 0
    else
        if [ "$required" = "true" ]; then
            echo -e "${RED}❌ FAILED${NC}"
            ERRORS=$((ERRORS + 1))
        else
            echo -e "${YELLOW}⚠️  WARNING${NC}"
            WARNINGS=$((WARNINGS + 1))
        fi
        return 1
    fi
}

# Prerequisites
echo -e "${BLUE}=== Prerequisites ===${NC}"
test_item "Globus CLI installed" "command -v globus"
test_item "Globus logged in" "globus whoami" false
test_item "Python 3 available" "python3 --version"
test_item "Conda/Mamba available" "command -v conda || command -v mamba" false
echo ""

# File Structure
echo -e "${BLUE}=== File Structure ===${NC}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_item "Repository root exists" "[ -d $REPO_ROOT ]"
test_item "Snakefile exists" "[ -f $REPO_ROOT/Snakefile ]"
test_item "Config file exists" "[ -f $REPO_ROOT/config/scinet.yaml ]"
test_item "Workflow script exists" "[ -f $REPO_ROOT/scripts/workflow.sh ]"
test_item "SLURM script exists" "[ -f $REPO_ROOT/slurm/run_snakemake.sh ]"
echo ""

# Python Package
echo -e "${BLUE}=== Python Package ===${NC}"
test_item "svs_raw_api package" "python3 -c 'import svs_raw_api'" false
test_item "numpy installed" "python3 -c 'import numpy'" false
test_item "yaml installed" "python3 -c 'import yaml'" false
test_item "pidng installed" "python3 -c 'import pidng'" false
echo ""

# Tools
echo -e "${BLUE}=== External Tools ===${NC}"
test_item "Snakemake installed" "snakemake --version" false
source "$REPO_ROOT/scripts/validate_rawtherapee.sh" &>/dev/null || true
source "$REPO_ROOT/scripts/rawtherapee_path.sh" &>/dev/null || true
if [ -n "$RT_CLI_PATH" ]; then
    test_item "RawTherapee CLI found" "[ -f $RT_CLI_PATH ]" false
else
    echo -e "Testing: RawTherapee CLI found ... ${YELLOW}⚠️  WARNING${NC}"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# Configuration
echo -e "${BLUE}=== Configuration ===${NC}"
test_item "Config loads correctly" "python3 $REPO_ROOT/tests/test_config.py" false
echo ""

# Storage (if on Ceres)
if hostname | grep -q "ceres"; then
    echo -e "${BLUE}=== Storage Access ===${NC}"
    test_item "JUNO archive accessible" "[ -d /project/dash_agir/semifield-upload ]" false
    test_item "Ceres scratch accessible" "[ -d /90daydata/dash_agir/data ]" false
    test_item "Project directory accessible" "[ -d /project/dash_agir/matthew.kutugata ]" false
    echo ""
    
    # SLURM
    echo -e "${BLUE}=== SLURM ===${NC}"
    test_item "SLURM commands available" "command -v squeue" false
    test_item "Can query SLURM" "squeue -u $USER -A dash_agir" false
    echo ""
fi

# Summary
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  ${WARNINGS} warnings (non-critical)${NC}"
else
    echo -e "${RED}❌ ${ERRORS} errors, ${WARNINGS} warnings${NC}"
fi
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

if [ $ERRORS -gt 0 ]; then
    echo "Run setup script to fix errors:"
    echo "  bash scripts/setup_environment.sh"
    exit 1
fi

exit 0
