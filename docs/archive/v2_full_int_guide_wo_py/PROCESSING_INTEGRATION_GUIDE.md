# SVS RAW Processing Pipeline - Complete Integration Guide

## Overview

This guide integrates your `svs_raw_api` image processing code into the three-tier pipeline infrastructure for automated RAW → DNG → JPG conversion on SCINet Ceres.

**Complete Pipeline:**
```
NCSU NFS → JUNO Archive → Ceres Scratch → RAW→DNG→JPG Processing → JUNO Archive
```

## What You Have Now

✅ **Three-tier pipeline infrastructure:**
- Database tracking (db_manager.py)
- Globus transfer management (globus_manager.py)
- Workflow orchestration (workflow.sh)

✅ **Image processing code:**
- `svs_raw_api` Python package
- RAW → DNG conversion (SVSRaw2DNG)
- DNG → JPG conversion (RawTherapee)

## What's New

🆕 **Snakemake Workflow:**
- Parallel processing of images
- Automatic task distribution across SLURM jobs
- Resource management and error handling

🆕 **SLURM Integration:**
- Automated job submission
- Database status tracking
- Log file management

## Files to Install

All files are in `/mnt/user-data/outputs/`:

1. **Snakefile** - Snakemake workflow for parallel RAW processing
2. **snakemake_config.yaml** - Configuration file
3. **run_snakemake.sh** - SLURM submission script
4. **workflow_integrated.sh** - Updated workflow manager

## Installation Steps

### Step 1: Set Up Directory Structure

```bash
# On Ceres login node
cd ~/repos/svs-raw-api

# Create directories
mkdir -p config
mkdir -p slurm
mkdir -p data/profiles
mkdir -p logs
```

### Step 2: Copy New Files

```bash
# From your local machine (where you downloaded the files)
scp Snakefile matthew.kutugata@ceres-dtn:~/repos/svs-raw-api/
scp snakemake_config.yaml matthew.kutugata@ceres-dtn:~/repos/svs-raw-api/config/
scp run_snakemake.sh matthew.kutugata@ceres-dtn:~/repos/svs-raw-api/slurm/
scp workflow_integrated.sh matthew.kutugata@ceres-dtn:~/repos/svs-raw-api/scripts/workflow.sh

# Make scripts executable
ssh matthew.kutugata@ceres-dtn "chmod +x ~/repos/svs-raw-api/slurm/run_snakemake.sh"
ssh matthew.kutugata@ceres-dtn "chmod +x ~/repos/svs-raw-api/scripts/workflow.sh"
```

### Step 3: Verify Configuration Files

You need these files in `data/profiles/`:

```bash
# On Ceres
cd ~/repos/svs-raw-api/data/profiles

# Check for:
ls -lh svs_tags.yaml
ls -lh MD_calibration_matrix_optimized.npy
ls -lh MD_shr661_raw16.pp3
```

If missing, copy from your existing setup or create them.

### Step 4: Update Configuration Paths

Edit `config/snakemake_config.yaml` to match your paths:

```yaml
paths:
  repo_root: /home/matthew.kutugata/repos/svs-raw-api
  ceres_scratch: /90daydata/dash_agir/data/semifield-upload
  output_base: /project/dash_agir/matthew.kutugata/semifield-developed-images
  
  svs_tags: /home/matthew.kutugata/repos/svs-raw-api/data/profiles/svs_tags.yaml
  color_matrix: /home/matthew.kutugata/repos/svs-raw-api/data/profiles/MD_calibration_matrix_optimized.npy
  pp3_profile: /home/matthew.kutugata/repos/svs-raw-api/data/profiles/MD_shr661_raw16.pp3
  rawtherapee_cli: /home/matthew.kutugata/SemiF-Preprocessing/squashfs-root/usr/bin/rawtherapee-cli

processing:
  height: 3024
  width: 4032
  threads_per_image: 4
```

### Step 5: Install svs_raw_api Package

```bash
# On Ceres compute node
salloc -A dash_agir -t 01:00:00 -c 4

# Activate conda environment
source /project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep/bin/activate

# Install package
cd ~/repos/svs-raw-api
pip install -e . --no-deps

# Verify installation
python -c "from svs_raw_api import SVSRaw2DNG; print('✅ svs_raw_api installed')"

# Exit compute node
exit
```

### Step 6: Install Snakemake

```bash
# On compute node with your conda env activated
pip install snakemake --break-system-packages

# Verify
snakemake --version
```

## Testing the Setup

### Test 1: Verify Files and Paths

```bash
cd ~/repos/svs-raw-api

# Check all required files exist
echo "Checking files..."
[ -f "Snakefile" ] && echo "✅ Snakefile" || echo "❌ Snakefile missing"
[ -f "config/snakemake_config.yaml" ] && echo "✅ Config" || echo "❌ Config missing"
[ -f "slurm/run_snakemake.sh" ] && echo "✅ SLURM script" || echo "❌ SLURM script missing"
[ -f "scripts/workflow.sh" ] && echo "✅ Workflow" || echo "❌ Workflow missing"
```

### Test 2: Test Snakemake Dry Run

```bash
# Get a compute node
salloc -A dash_agir -t 01:00:00 -c 4

# Activate environment
source /project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep/bin/activate

# Navigate to repo
cd ~/repos/svs-raw-api

# Dry run (shows what would be done without actually doing it)
snakemake \
    --snakefile Snakefile \
    --configfile config/snakemake_config.yaml \
    --config batch_id=TEST_2025-01-01 mode=single \
    --dry-run \
    --printshellcmds

# If this works, Snakemake is configured correctly
exit
```

### Test 3: Process a Small Test Batch

For this test, you need a small batch (~5-10 images) already transferred to Ceres.

```bash
# Make sure a test batch exists on Ceres
ls /90daydata/dash_agir/data/semifield-upload/MD_2025-10-22/

# Process it with the workflow
cd ~/repos/svs-raw-api
./scripts/workflow.sh process MD_2025-10-22

# Monitor the job
watch -n 10 'squeue -u $USER -A dash_agir'

# Check output
ls -lh /project/dash_agir/matthew.kutugata/semifield-developed-images/MD_2025-10-22/images/
```

## Usage

### Complete Pipeline Workflow

```bash
# 1. Discover new batches
./scripts/workflow.sh check-missing

# 2. Sync from NCSU to JUNO (if needed)
./scripts/workflow.sh sync MD_2025-10-22

# 3. Transfer from JUNO to Ceres
./scripts/workflow.sh transfer MD_2025-10-22

# 4. Process on Ceres (RAW → DNG → JPG)
./scripts/workflow.sh process MD_2025-10-22

# OR: Run entire pipeline in one command
./scripts/workflow.sh full-pipeline MD_2025-10-22
```

### Monitoring Processing Jobs

```bash
# Check active SLURM jobs
squeue -u $USER -A dash_agir

# Check specific job status
sacct -j <job-id> --format=JobID,JobName,State,Elapsed

# View real-time log
tail -f /project/dash_agir/matthew.kutugata/logs/snakemake_svs_raw_process_<job-id>.out

# Check output images
ls -lh /project/dash_agir/matthew.kutugata/semifield-developed-images/<batch-id>/images/
```

### Batch Processing Multiple Batches

```bash
# Process all transferred batches
for batch in MD_2025-10-22 MD_2025-10-23 MD_2025-10-24; do
    ./scripts/workflow.sh process $batch
    sleep 30  # Stagger submissions
done

# Monitor all jobs
watch -n 30 'squeue -u $USER -A dash_agir'
```

## Performance

**Per 100-image batch:**
- NCSU → JUNO sync: 10-30 minutes (one-time)
- JUNO → Ceres transfer: 5-10 minutes
- **RAW → DNG → JPG processing: 20-30 minutes**
  - 12 parallel SLURM jobs
  - 4 cores × 16GB per job
  - ~2-3 minutes per image (RAW→DNG→JPG)

**Total first-time processing: ~40-60 minutes per batch**

## Resource Allocation

**Per Batch Processing Job:**
- **Cores:** 48 total (12 jobs × 4 cores each)
- **Memory:** 192GB total (12 jobs × 16GB each)
- **Time:** 4 hours (max, usually completes in 20-40 min)
- **Partition:** short (recommended)

## Output Structure

```
/project/dash_agir/matthew.kutugata/semifield-developed-images/
└── MD_2025-10-22/
    ├── dngs/              # DNG files (kept for archival)
    │   ├── MD_1234567890.dng
    │   └── ...
    ├── images/            # Final JPG files
    │   ├── MD_1234567890.jpg
    │   └── ...
    ├── logs/              # Processing logs per image
    │   ├── raw_to_dng_MD_1234567890.log
    │   ├── dng_to_jpg_MD_1234567890.log
    │   └── ...
    └── processing_summary.txt
```

## Troubleshooting

### "svs_raw_api not found"

```bash
# Reinstall package
cd ~/repos/svs-raw-api
source /project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep/bin/activate
pip install -e . --no-deps
```

### "RawTherapee CLI not found"

```bash
# Check if it exists
ls -lh /home/matthew.kutugata/SemiF-Preprocessing/squashfs-root/usr/bin/rawtherapee-cli

# Update path in config/snakemake_config.yaml if different
```

### Processing Job Fails

```bash
# Check SLURM output
cat /project/dash_agir/matthew.kutugata/logs/snakemake_svs_raw_process_<job-id>.out

# Check Snakemake logs
cat /project/dash_agir/matthew.kutugata/logs/snakemake_<batch-id>_<job-id>.log

# Check individual image logs
cat /project/dash_agir/matthew.kutugata/semifield-developed-images/<batch-id>/logs/raw_to_dng_*.log
```

### "Batch not found in scratch"

Make sure to transfer first:
```bash
./scripts/workflow.sh transfer MD_2025-10-22
```

### Out of Memory Errors

Reduce parallel jobs in `config/snakemake_config.yaml`:
```yaml
slurm:
  max_parallel_jobs: 8  # Reduce from 12
```

## Advanced Usage

### Process Specific Images Only

Edit Snakefile to filter by pattern:
```python
RAW_FILES = [f for f in INPUT_DIR.glob("*.ARW") if "subset" in f.name]
```

### Custom PP3 Profile

```bash
# Create custom profile for different states/lighting
cp data/profiles/MD_shr661_raw16.pp3 data/profiles/NC_custom.pp3

# Edit NC_custom.pp3 with your settings

# Update config for specific batch
sed -i 's/MD_shr661_raw16.pp3/NC_custom.pp3/' config/snakemake_config.yaml
```

### Cleanup Intermediate Files

To save space, enable cleanup in config:
```yaml
processing:
  cleanup_dngs: true     # Delete DNGs after JPG creation
  cleanup_raw: false     # Keep RAW in scratch (auto-deleted after 90 days)
```

## Database Integration

Processing status is automatically tracked:

```sql
-- Check processing status
SELECT batch_id, processing_status, processing_started_at, processing_completed_at
FROM batches
WHERE processing_status = 'completed'
ORDER BY processing_completed_at DESC;

-- View processing history
SELECT * FROM processing_history
WHERE batch_id = 'MD_2025-10-22';
```

## Directory Permissions

Ensure proper permissions:

```bash
# Logs directory
mkdir -p /project/dash_agir/matthew.kutugata/logs
chmod 755 /project/dash_agir/matthew.kutugata/logs

# Output directory
mkdir -p /project/dash_agir/matthew.kutugata/semifield-developed-images
chmod 755 /project/dash_agir/matthew.kutugata/semifield-developed-images

# Scratch directory (managed by system)
# /90daydata/dash_agir/data/semifield-upload
```

## Next Steps

1. ✅ Install all files
2. ✅ Update configuration paths
3. ✅ Install svs_raw_api package
4. ✅ Install Snakemake
5. ✅ Test with small batch
6. ✅ Process real batches
7. ✅ Monitor and optimize

## Support

- **Pipeline Issues:** Check THREE_TIER_SETUP.md
- **Processing Issues:** Check logs in output directory
- **SLURM Issues:** Check SLURM output in logs directory
- **Snakemake Help:** https://snakemake.readthedocs.io/

## Quick Reference

```bash
# Complete pipeline
./scripts/workflow.sh full-pipeline <BATCH_ID>

# Just processing (batch already on Ceres)
./scripts/workflow.sh process <BATCH_ID>

# Check status
./scripts/workflow.sh status

# Monitor jobs
squeue -u $USER -A dash_agir

# View outputs
ls /project/dash_agir/matthew.kutugata/semifield-developed-images/<BATCH_ID>/images/
```
