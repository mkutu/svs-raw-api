# Three-Tier Pipeline - Complete File List

## Files Created

All files are available in `/mnt/user-data/outputs/`

### 1. Core Scripts (Updated Versions)

| File | Purpose | Install To |
|------|---------|------------|
| `globus_manager_v2.py` | Enhanced Globus manager with NCSU support | `~/repos/svs-raw-api/scripts/globus_manager.py` |
| `db_manager_v2.py` | Enhanced database manager with NCSU tracking | `~/repos/svs-raw-api/scripts/db_manager.py` |
| `workflow_v2.sh` | Enhanced workflow with three-tier commands | `~/repos/svs-raw-api/scripts/workflow.sh` |

### 2. New Utilities

| File | Purpose | Install To |
|------|---------|------------|
| `find_ncsu_endpoint.sh` | Interactive NCSU endpoint finder | `~/repos/svs-raw-api/scripts/find_ncsu_endpoint.sh` |

### 3. Documentation

| File | Purpose | Install To |
|------|---------|------------|
| `THREE_TIER_SETUP.md` | Complete setup and configuration guide | `~/repos/svs-raw-api/docs/THREE_TIER_SETUP.md` |
| `THREE_TIER_QUICK_REF.md` | Quick reference card for commands | `~/repos/svs-raw-api/docs/THREE_TIER_QUICK_REF.md` |
| `THREE_TIER_SUMMARY.md` | What changed and how to migrate | `~/repos/svs-raw-api/docs/THREE_TIER_SUMMARY.md` |

## Installation Script

Save this as `install_three_tier.sh` and run it:

```bash
#!/bin/bash
# Three-Tier Pipeline Installation Script

set -e

REPO_ROOT="$HOME/repos/svs-raw-api"
SOURCE_DIR="/mnt/user-data/outputs"

echo "Installing Three-Tier Pipeline Enhancement..."
echo ""

# Backup existing files
echo "1. Backing up existing files..."
mkdir -p "$REPO_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
cp "$REPO_ROOT/scripts/globus_manager.py" "$REPO_ROOT/backups/$(date +%Y%m%d_%H%M%S)/" 2>/dev/null || true
cp "$REPO_ROOT/scripts/db_manager.py" "$REPO_ROOT/backups/$(date +%Y%m%d_%H%M%S)/" 2>/dev/null || true
cp "$REPO_ROOT/scripts/workflow.sh" "$REPO_ROOT/backups/$(date +%Y%m%d_%H%M%S)/" 2>/dev/null || true
echo "   ✓ Backups created"

# Create docs directory if needed
mkdir -p "$REPO_ROOT/docs"
mkdir -p "$REPO_ROOT/config"
mkdir -p "$REPO_ROOT/slurm"

# Install core scripts
echo ""
echo "2. Installing core scripts..."
cp "$SOURCE_DIR/globus_manager_v2.py" "$REPO_ROOT/scripts/globus_manager.py"
cp "$SOURCE_DIR/db_manager_v2.py" "$REPO_ROOT/scripts/db_manager.py"
cp "$SOURCE_DIR/workflow_v2.sh" "$REPO_ROOT/scripts/workflow.sh"
echo "   ✓ Core scripts installed"

# Install Snakemake files
echo ""
echo "3. Installing Snakemake files..."
cp "$SOURCE_DIR/Snakefile_three_tier" "$REPO_ROOT/"
cp "$SOURCE_DIR/snakemake_config_three_tier.yaml" "$REPO_ROOT/config/"
cp "$SOURCE_DIR/run_snakemake_three_tier.sh" "$REPO_ROOT/slurm/"
echo "   ✓ Snakemake files installed"

# Install utilities
echo ""
echo "4. Installing utilities..."
cp "$SOURCE_DIR/find_ncsu_endpoint.sh" "$REPO_ROOT/scripts/"
echo "   ✓ Utilities installed"

# Install documentation
echo ""
echo "5. Installing documentation..."
cp "$SOURCE_DIR/README.md" "$REPO_ROOT/docs/THREE_TIER_README.md"
cp "$SOURCE_DIR/THREE_TIER_SETUP.md" "$REPO_ROOT/docs/"
cp "$SOURCE_DIR/THREE_TIER_QUICK_REF.md" "$REPO_ROOT/docs/"
cp "$SOURCE_DIR/THREE_TIER_SUMMARY.md" "$REPO_ROOT/docs/"
cp "$SOURCE_DIR/SNAKEMAKE_INTEGRATION.md" "$REPO_ROOT/docs/"
cp "$SOURCE_DIR/ARCHITECTURE_DIAGRAM.txt" "$REPO_ROOT/docs/"
echo "   ✓ Documentation installed"

# Make scripts executable
echo ""
echo "6. Setting permissions..."
chmod +x "$REPO_ROOT/scripts/"*.sh
chmod +x "$REPO_ROOT/slurm/"*.sh
echo "   ✓ Permissions set"

# Verify installation
echo ""
echo "7. Verifying installation..."
if [ -f "$REPO_ROOT/scripts/globus_manager.py" ] && \
   [ -f "$REPO_ROOT/scripts/db_manager.py" ] && \
   [ -f "$REPO_ROOT/scripts/workflow.sh" ] && \
   [ -f "$REPO_ROOT/scripts/find_ncsu_endpoint.sh" ] && \
   [ -f "$REPO_ROOT/Snakefile_three_tier" ] && \
   [ -f "$REPO_ROOT/slurm/run_snakemake_three_tier.sh" ]; then
    echo "   ✓ All files installed correctly"
else
    echo "   ✗ Some files missing - check installation"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Installation complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure NCSU endpoint:"
echo "   cd $REPO_ROOT"
echo "   ./scripts/find_ncsu_endpoint.sh"
echo ""
echo "2. Update Snakemake config (if needed):"
echo "   nano config/snakemake_config_three_tier.yaml"
echo ""
echo "3. Test discovery:"
echo "   ./scripts/workflow.sh check-missing"
echo ""
echo "4. Read documentation:"
echo "   cat docs/THREE_TIER_SETUP.md"
echo "   cat docs/SNAKEMAKE_INTEGRATION.md"
echo ""
echo "For help:"
echo "   ./scripts/workflow.sh help"
echo ""
