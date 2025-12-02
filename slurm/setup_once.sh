#!/bin/bash
# slurm/setup_once.sh
# Run this ONCE to set up your environment on SciNet
# Usage: bash slurm/setup_once.sh

set -e  # Exit on error

echo "================================================"
echo "SVS RAW API - One-Time Setup for SciNet"
echo "================================================"

# Define paths
REPO_DIR="$HOME/repos/svs-raw-api"
PROJECT_BASE="/project/dash_agir/matthew.kutugata"
ENV_PATH="$PROJECT_BASE/software/miniforge3/envs/semif_prep"

cd $REPO_DIR

# 1. Create necessary directories in project space
echo "[1/5] Creating project directories..."
mkdir -p "$PROJECT_BASE/semifield-developed-images"
mkdir -p "$PROJECT_BASE/logs"
mkdir -p "$PROJECT_BASE/software"
echo "✓ Directories created"

# 2. Validate RawTherapee (run once to cache result)
echo "[2/5] Validating RawTherapee..."
bash scripts/validate_rawtherapee.sh > scripts/rawtherapee_path.sh
source scripts/rawtherapee_path.sh
echo "✓ RawTherapee validated: $RAWTHERAPEE_CLI"

# Update config with actual RawTherapee path
if [ -f "conf/scinet.yaml" ]; then
    # Update the rawtherapee_cli path in config
    sed -i "s|rawtherapee_cli:.*|rawtherapee_cli: $RAWTHERAPEE_CLI|" conf/scinet.yaml
    echo "✓ Config updated with RawTherapee path"
fi

# 3. Check conda environment
echo "[3/5] Checking conda environment..."
module load miniconda
source activate $ENV_PATH
echo "✓ Environment created"

# 4. Install package in editable mode
echo "[4/5] Installing svs-raw-api package..."
source activate $ENV_PATH
pip install -e .
echo "✓ Package installed"

# 5. Test configuration
echo "[5/5] Testing configuration..."
python -c "
from pathlib import Path
import yaml
config_path = Path('conf/scinet.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)
print('✓ Config loaded successfully')
print('Output dir: ', config['paths']['output_dir']) 
print('Logs dir: ', config['paths']['logs_dir'])
"

echo ""
echo "================================================"
echo "Setup complete! You can now submit jobs with:"
echo "  sbatch slurm/process_batch.sh <input_dir> <date>"
echo "or"
echo "  sbatch slurm/array_job.sh"
echo "================================================"