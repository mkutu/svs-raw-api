#!/bin/bash
# ============================================================================
# Environment Setup Script for SVS RAW Processing Pipeline
# ============================================================================
#
# This script helps set up the processing environment on SciNet Ceres
#
# Usage:
#   ./scripts/setup_environment.sh
#
# ============================================================================

set -e

echo "============================================================================"
echo "SVS RAW Processing Pipeline - Environment Setup"
echo "============================================================================"
echo ""

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# ============================================================================
# Step 1: Check Prerequisites
# ============================================================================

echo "[1/5] Checking prerequisites..."
echo ""

# Check Python
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo "✓ Python $PYTHON_VERSION found"
else
    echo "✗ Python not found"
    echo "  Please activate a conda environment first"
    exit 1
fi

# Check conda
if command -v conda &> /dev/null; then
    echo "✓ Conda found"
else
    echo "⚠ Conda not found (optional)"
fi

# Check SLURM
if command -v sbatch &> /dev/null; then
    echo "✓ SLURM found"
else
    echo "✗ SLURM not found"
    echo "  This script must be run on a SLURM cluster (e.g., SciNet Ceres)"
    exit 1
fi

echo ""

# ============================================================================
# Step 2: Install Python Package
# ============================================================================

echo "[2/5] Installing svs-raw-api package..."
echo ""

pip install -e . --no-deps

# Verify installation
if python -c "from svs_raw_api import SVSRaw2DNG" 2>/dev/null; then
    echo "✓ svs-raw-api package installed successfully"
else
    echo "✗ Package installation failed"
    exit 1
fi

echo ""

# ============================================================================
# Step 3: Install Snakemake
# ============================================================================

echo "[3/5] Checking Snakemake installation..."
echo ""

if command -v snakemake &> /dev/null; then
    SNAKE_VERSION=$(snakemake --version)
    echo "✓ Snakemake $SNAKE_VERSION already installed"
else
    echo "Installing Snakemake..."
    pip install snakemake
    echo "✓ Snakemake installed"
fi

echo ""

# ============================================================================
# Step 4: Check RawTherapee
# ============================================================================

echo "[4/5] Checking RawTherapee..."
echo ""

if [ -f "$REPO_DIR/scripts/rawtherapee_path.sh" ]; then
    source "$REPO_DIR/scripts/rawtherapee_path.sh"
    if [ -f "$RAWTHERAPEE_CLI" ]; then
        echo "✓ RawTherapee CLI found: $RAWTHERAPEE_CLI"
    else
        echo "⚠ RawTherapee CLI not found at: $RAWTHERAPEE_CLI"
        echo "  Please update config/config.yaml with correct path"
    fi
else
    echo "⚠ RawTherapee path script not found"
    echo "  Please ensure RawTherapee is installed and update config/config.yaml"
fi

echo ""

# ============================================================================
# Step 5: Create Directories
# ============================================================================

echo "[5/5] Creating directories..."
echo ""

# Create log directory
LOG_DIR="/project/dash_agir/matthew.kutugata/logs"
mkdir -p "$LOG_DIR" 2>/dev/null || echo "⚠ Could not create $LOG_DIR (may already exist)"

# Create output directory
OUTPUT_DIR="/project/dash_agir/matthew.kutugata/semifield-developed-images"
mkdir -p "$OUTPUT_DIR" 2>/dev/null || echo "⚠ Could not create $OUTPUT_DIR (may already exist)"

echo "✓ Directory structure verified"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "============================================================================"
echo "Setup Complete!"
echo "============================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure paths in config/config.yaml:"
echo "   - repo_root"
echo "   - rawtherapee_cli"
echo "   - output_dir"
echo ""
echo "2. Test with a dry run:"
echo "   snakemake --config batch_id=TEST --dry-run"
echo ""
echo "3. Process a batch:"
echo "   ./scripts/process_batch.sh MD_2025-10-22"
echo ""
echo "For help, see: docs/QUICKSTART.md"
echo "============================================================================"
