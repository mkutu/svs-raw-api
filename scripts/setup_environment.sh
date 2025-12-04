#!/bin/bash
#
# SVS RAW API - Environment Setup Script
# One-time setup for USDA SCINet Ceres HPC
#
# Usage: bash scripts/setup_environment.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}         SVS RAW API - Environment Setup                     ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_BASE="/project/dash_agir/matthew.kutugata"
CONDA_ENV="$PROJECT_BASE/software/miniforge3/envs/semif_prep"

cd "$REPO_ROOT"

# ============================================================================
# Step 1: Create project directories
# ============================================================================

echo -e "${BLUE}[1/6] Creating project directories...${NC}"

DIRS=(
    "$PROJECT_BASE/semifield-developed-images"
    "$PROJECT_BASE/logs"
    "$PROJECT_BASE/software"
    "$REPO_ROOT/data/profiles"
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "  ✓ Created: $dir"
    else
        echo "  ✓ Exists: $dir"
    fi
done

echo -e "${GREEN}✓ Directories ready${NC}"
echo ""

# ============================================================================
# Step 2: Validate RawTherapee
# ============================================================================

echo -e "${BLUE}[2/6] Validating RawTherapee...${NC}"

bash scripts/validate_rawtherapee.sh > scripts/rawtherapee_path.sh
source scripts/rawtherapee_path.sh

if [ -f "$RT_CLI_PATH" ]; then
    echo -e "${GREEN}✓ RawTherapee found: $RT_CLI_PATH${NC}"
    
    # Update config with actual path
    if [ -f "config/scinet.yaml" ]; then
        sed -i "s|rawtherapee_cli:.*|rawtherapee_cli: $RT_CLI_PATH|" config/scinet.yaml
        echo "  ✓ Config updated"
    fi
else
    echo -e "${RED}✗ RawTherapee not found${NC}"
    echo "  Please install RawTherapee and update config/scinet.yaml"
fi

echo ""

# ============================================================================
# Step 3: Check conda environment
# ============================================================================

echo -e "${BLUE}[3/6] Checking conda environment...${NC}"

if [ -d "$CONDA_ENV" ]; then
    echo -e "${GREEN}✓ Environment exists: $CONDA_ENV${NC}"
else
    echo -e "${YELLOW}⚠  Environment not found: $CONDA_ENV${NC}"
    echo "  Creating environment..."
    
    module load miniconda
    conda create -p "$CONDA_ENV" python=3.12 -y
    echo -e "${GREEN}✓ Environment created${NC}"
fi

echo ""

# ============================================================================
# Step 4: Install Python package
# ============================================================================

echo -e "${BLUE}[4/6] Installing svs-raw-api package...${NC}"

module load miniconda
source activate "$CONDA_ENV"

# Install in editable mode
pip install -e . --no-deps

if python -c "import svs_raw_api" 2>/dev/null; then
    echo -e "${GREEN}✓ Package installed successfully${NC}"
else
    echo -e "${RED}✗ Package installation failed${NC}"
    exit 1
fi

echo ""

# ============================================================================
# Step 5: Install Snakemake
# ============================================================================

echo -e "${BLUE}[5/6] Installing Snakemake...${NC}"

if command -v snakemake &> /dev/null; then
    echo -e "${GREEN}✓ Snakemake already installed${NC}"
    snakemake --version
else
    echo "  Installing Snakemake..."
    pip install snakemake --break-system-packages
    echo -e "${GREEN}✓ Snakemake installed${NC}"
fi

echo ""

# ============================================================================
# Step 6: Verify setup
# ============================================================================

echo -e "${BLUE}[6/6] Verifying setup...${NC}"

# Test config loading
python -c "
from pathlib import Path
import yaml

config_path = Path('config/scinet.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)
    
print('  ✓ Config loaded')
print(f'    Output dir: {config[\"paths\"][\"output_base\"]}')
print(f'    Logs dir: {Path(config[\"paths\"][\"output_base\"]).parent / \"logs\"}')
"

# Check Globus
if command -v globus &> /dev/null; then
    if globus whoami &> /dev/null; then
        echo "  ✓ Globus authenticated"
    else
        echo -e "  ${YELLOW}⚠  Globus not authenticated. Run: globus login${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠  Globus CLI not installed. Install: pip install globus-cli${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Setup complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
echo "  1. Configure Globus endpoints (if needed):"
echo "     bash scripts/find_ncsu_endpoint.sh"
echo ""
echo "  2. Test the configuration:"
echo "     bash scripts/test_setup.sh"
echo ""
echo "  3. Process a batch:"
echo "     ./scripts/workflow.sh full-pipeline <BATCH_ID>"
echo ""
