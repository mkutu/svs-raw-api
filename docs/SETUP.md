# Setup Guide

Complete installation and configuration guide for the SVS RAW Processing Pipeline on USDA SCINet Ceres HPC.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Verification](#verification)
5. [First Run](#first-run)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Access

- **USDA SCINet Account**: Active Ceres HPC account
- **SLURM Account**: `dash_agir` allocation
- **Globus Account**: For data transfers between storage tiers
- **Storage Access**:
  - JUNO LTS: `/project/dash_agir/semifield-upload`
  - Ceres Scratch: `/90daydata/dash_agir/data`
  - Project Space: `/project/dash_agir/matthew.kutugata`

### Software Requirements

- Python 3.8+
- Conda/Miniconda (available via `module load miniconda`)
- Globus CLI
- RawTherapee CLI (for DNG → JPG conversion)
- Snakemake 7.0+

## Installation

### Step 1: Clone Repository

```bash
# Login to Ceres
ssh matthew.kutugata@ceres.scinet.usda.gov

# Navigate to repos directory
mkdir -p ~/repos
cd ~/repos

# Clone repository
git clone <repository-url> svs-raw-api
cd svs-raw-api
```

### Step 2: Run Automated Setup

The automated setup script will:
- Create necessary directories
- Validate RawTherapee installation
- Setup conda environment
- Install Python package
- Install Snakemake

```bash
bash scripts/setup_environment.sh
```

This script is idempotent and can be run multiple times safely.

### Step 3: Manual Environment Setup (Alternative)

If you prefer manual setup:

```bash
# Load miniconda
module load miniconda

# Create or activate environment
CONDA_ENV="/project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep"
conda create -p $CONDA_ENV python=3.12 -y
source activate $CONDA_ENV

# Install package
pip install -e .

# Install Snakemake
pip install snakemake --break-system-packages

# Install additional tools
pip install globus-cli pyyaml numpy pidng
```

## Configuration

### Config File Structure

The pipeline uses `config/scinet.yaml` for all configuration:

```yaml
paths:
  # Repository and project
  repo_root: /home/matthew.kutugata/repos/svs-raw-api
  project_base: /project/dash_agir/matthew.kutugata
  
  # Storage tiers
  ceres_scratch: /90daydata/dash_agir/data/semifield-upload
  output_base: ${project_base}/semifield-developed-images
  
  # Processing profiles
  svs_tags: ${repo_root}/data/profiles/svs_tags.yaml
  color_matrix: ${repo_root}/data/profiles/MD_calibration_matrix_optimized.npy
  pp3_profile: ${repo_root}/data/profiles/MD_shr661_raw16.pp3
  rawtherapee_cli: /path/to/rawtherapee-cli

processing:
  height: 3024
  width: 4032
  threads_per_image: 4
  cleanup_dngs: false

slurm:
  partition: short
  account: dash_agir
  time: "04:00:00"
  mem_per_cpu: 4GB
  cpus_per_task: 4
  max_parallel_jobs: 12
```

### Critical Files to Configure

1. **RawTherapee Path**: Auto-detected by `validate_rawtherapee.sh`
2. **Processing Profiles**: Place in `data/profiles/`
   - `svs_tags.yaml` - Camera metadata
   - `MD_calibration_matrix_optimized.npy` - Color calibration matrix
   - `MD_shr661_raw16.pp3` - RawTherapee processing profile

### Globus Configuration

```bash
# Login to Globus
globus login

# Verify authentication
globus whoami

# Find your NCSU endpoint (if using three-tier pipeline)
bash scripts/find_ncsu_endpoint.sh
```

## Verification

### Automated Test

```bash
cd ~/repos/svs-raw-api
bash scripts/test_setup.sh
```

This will check:
- ✅ Globus authentication
- ✅ Conda environment
- ✅ Python packages
- ✅ RawTherapee CLI
- ✅ Configuration files
- ✅ Storage access
- ✅ SLURM access

### Manual Verification

```bash
# 1. Check Python package
python -c "import svs_raw_api; print(svs_raw_api.__version__)"

# 2. Check Snakemake
snakemake --version

# 3. Check RawTherapee
source scripts/rawtherapee_path.sh
$RT_CLI_PATH -v

# 4. Check storage access
ls /project/dash_agir/semifield-upload/
ls /90daydata/dash_agir/data/semifield-upload/

# 5. Check SLURM
squeue -u $USER -A dash_agir
```

## First Run

### Test with Small Batch

1. **Ensure test batch exists on Ceres**:
   ```bash
   TEST_BATCH="MD_2025-10-22"  # Replace with actual batch
   ls /90daydata/dash_agir/data/semifield-upload/$TEST_BATCH/
   ```

2. **Run test processing**:
   ```bash
   cd ~/repos/svs-raw-api
   ./scripts/workflow.sh process $TEST_BATCH
   ```

3. **Monitor job**:
   ```bash
   # Watch job queue
   watch -n 10 'squeue -u $USER -A dash_agir'
   
   # Check logs
   tail -f /project/dash_agir/matthew.kutugata/logs/snakemake_*.out
   ```

4. **Verify output**:
   ```bash
   # Check output directory
   ls /project/dash_agir/matthew.kutugata/semifield-developed-images/$TEST_BATCH/
   
   # Check processing summary
   cat /project/dash_agir/matthew.kutugata/semifield-developed-images/$TEST_BATCH/processing_summary.txt
   ```

### Expected Timeline

For a 100-image batch:
- **Job submission**: ~30 seconds
- **Environment setup**: ~2 minutes
- **RAW → DNG conversion**: ~10-15 minutes (parallel)
- **DNG → JPG conversion**: ~10-15 minutes (parallel)
- **Total**: ~25-35 minutes

## Troubleshooting

### Package Not Found

```bash
# Reinstall in editable mode
cd ~/repos/svs-raw-api
pip install -e . --no-deps
```

### RawTherapee Not Found

```bash
# Re-run validation
bash scripts/validate_rawtherapee.sh

# Check output
source scripts/rawtherapee_path.sh
echo $RT_CLI_PATH

# Manual specification
export RT_CLI_PATH="/path/to/your/rawtherapee-cli"
```

### Globus Issues

```bash
# Re-authenticate
globus logout
globus login

# Verify
globus whoami
globus endpoint search "NCSU"
```

### SLURM Job Failures

```bash
# Check job status
squeue -u $USER -A dash_agir

# View job details
scontrol show job <JOB_ID>

# Check logs
cat /project/dash_agir/matthew.kutugata/logs/snakemake_*_<JOB_ID>.err
```

### Snakemake Errors

```bash
# Dry run to test
snakemake --config batch_id=<BATCH_ID> --dry-run

# Verbose mode
snakemake --config batch_id=<BATCH_ID> --cores 4 --verbose

# Force rerun specific rule
snakemake --config batch_id=<BATCH_ID> --forcerun convert_raw_to_dng
```

### Permission Issues

```bash
# Check directory permissions
ls -ld /project/dash_agir/matthew.kutugata/
ls -ld /90daydata/dash_agir/data/

# Fix if needed (contact admin if you lack access)
```

## Next Steps

After successful setup:

1. **Review [Usage Guide](USAGE.md)** for detailed workflow examples
2. **Understand [Architecture](ARCHITECTURE.md)** for system design
3. **Process production batches** using the full pipeline
4. **Monitor and optimize** resource allocation

## Getting Help

- **Issues**: Open GitHub issue with:
  - Error messages
  - SLURM job ID
  - Config file (sanitized)
  - Output logs

- **SCINet Support**: scinet_vrsc@usda.gov
- **Pipeline Author**: matthew.kutugata@usda.gov
