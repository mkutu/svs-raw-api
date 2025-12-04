# Usage Guide

Detailed examples and workflows for the SVS RAW Processing Pipeline.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Basic Workflows](#basic-workflows)
3. [Advanced Usage](#advanced-usage)
4. [Batch Processing](#batch-processing)
5. [Monitoring](#monitoring)
6. [Performance Tuning](#performance-tuning)

## Quick Reference

### Common Commands

```bash
# Full pipeline (sync → transfer → process)
./scripts/workflow.sh full-pipeline <BATCH_ID>

# Individual stages
./scripts/workflow.sh sync <BATCH_ID>        # NCSU → JUNO
./scripts/workflow.sh transfer <BATCH_ID>    # JUNO → Ceres
./scripts/workflow.sh process <BATCH_ID>     # RAW → DNG → JPG

# Status and monitoring
./scripts/workflow.sh status                  # Pipeline status
squeue -u $USER -A dash_agir                 # Active jobs
sacct -X -u $USER --starttime=now-1day       # Recent jobs

# Direct SLURM submission
sbatch slurm/run_snakemake.sh <BATCH_ID>

# Snakemake directly
snakemake --config batch_id=<BATCH_ID> --cores 4
```

## Basic Workflows

### Workflow 1: Process Batch Already on Ceres

If your batch is already in Ceres scratch storage:

```bash
# Check batch exists
BATCH_ID="MD_2025-10-22"
ls /90daydata/dash_agir/data/semifield-upload/$BATCH_ID/

# Process it
cd ~/repos/svs-raw-api
./scripts/workflow.sh process $BATCH_ID

# Monitor (updates every 10 seconds)
watch -n 10 'squeue -u $USER -A dash_agir'

# Check output
ls /project/dash_agir/matthew.kutugata/semifield-developed-images/$BATCH_ID/images/
```

### Workflow 2: Complete Three-Tier Pipeline

For new data from NCSU:

```bash
BATCH_ID="MD_2025-10-23"

# Step 1: Check what needs syncing
./scripts/workflow.sh check-missing

# Step 2: Run full pipeline
./scripts/workflow.sh full-pipeline $BATCH_ID

# This will:
# 1. Sync NCSU → JUNO (if needed)
# 2. Transfer JUNO → Ceres scratch
# 3. Submit processing job
# 4. Process RAW → DNG → JPG
```

### Workflow 3: Manual Step-by-Step

For more control over each stage:

```bash
BATCH_ID="MD_2025-10-24"

# Stage 1: NCSU → JUNO (one-time per batch)
./scripts/workflow.sh sync $BATCH_ID
# Wait for Globus transfer: ~10-30 minutes

# Stage 2: JUNO → Ceres scratch
./scripts/workflow.sh transfer $BATCH_ID
# Wait for transfer: ~5-10 minutes

# Stage 3: Process on Ceres
./scripts/workflow.sh process $BATCH_ID
# Processing: ~20-40 minutes for 100 images
```

## Advanced Usage

### Custom Snakemake Options

```bash
BATCH_ID="MD_2025-10-25"

# Dry run (see what would be done)
snakemake --config batch_id=$BATCH_ID --dry-run --printshellcmds

# Run with specific cores
snakemake --config batch_id=$BATCH_ID --cores 8

# Force rerun specific rule
snakemake --config batch_id=$BATCH_ID --forcerun convert_raw_to_dng

# Generate workflow diagram
snakemake --config batch_id=$BATCH_ID --dag | dot -Tpng > workflow.png

# Detailed logging
snakemake --config batch_id=$BATCH_ID --verbose --printshellcmds
```

### Direct Python API

```python
#!/usr/bin/env python3
"""
Direct use of SVS RAW API for custom processing
"""

import numpy as np
from pathlib import Path
from svs_raw_api import SVSRaw2DNG

# Configuration
batch_id = "MD_2025-10-26"
input_dir = Path(f"/90daydata/dash_agir/data/semifield-upload/{batch_id}")
output_dir = Path(f"/project/dash_agir/matthew.kutugata/semifield-developed-images/{batch_id}/dngs")
output_dir.mkdir(parents=True, exist_ok=True)

# Load calibration matrix
color_matrix = np.load("data/profiles/MD_calibration_matrix_optimized.npy")

# Create converter
converter = SVSRaw2DNG(
    color_matrix=color_matrix,
    height=3024,
    width=4032
)

# Process all RAW files
for raw_file in input_dir.glob("*.RAW"):
    print(f"Processing {raw_file.name}...")
    
    # Load RAW image
    raw_data = np.fromfile(raw_file, dtype=np.uint16)
    raw_image = raw_data.reshape((3024, 4032))
    
    # Convert to DNG
    output_path = output_dir / f"{raw_file.stem}.dng"
    converter.save_dng(raw_image, output_path)
    
    print(f"  → Created {output_path.name}")
```

### Custom SLURM Parameters

Edit `slurm/run_snakemake.sh` for different resources:

```bash
# For larger batches (200+ images)
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=96
#SBATCH --mem=384G

# For smaller batches (<50 images)
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
```

Or submit with custom options:

```bash
sbatch --time=06:00:00 --cpus-per-task=64 slurm/run_snakemake.sh $BATCH_ID
```

## Batch Processing

### Process Multiple Batches

```bash
# List of batches to process
BATCHES=(
    "MD_2025-10-20"
    "MD_2025-10-21"
    "MD_2025-10-22"
    "MD_2025-10-23"
    "MD_2025-10-24"
)

# Process each batch (stagger submissions)
for batch in "${BATCHES[@]}"; do
    echo "Processing $batch..."
    ./scripts/workflow.sh process $batch
    sleep 30  # Wait 30 seconds between submissions
done

# Monitor all jobs
watch -n 15 'squeue -u $USER -A dash_agir --format="%.10i %.30j %.8T %.10M %.6D"'
```

### Process All Available Batches

```bash
# Find all batches in scratch that haven't been processed
SCRATCH_DIR="/90daydata/dash_agir/data/semifield-upload"
OUTPUT_DIR="/project/dash_agir/matthew.kutugata/semifield-developed-images"

for batch_dir in $SCRATCH_DIR/*/; do
    batch_id=$(basename $batch_dir)
    
    # Skip if already processed
    if [ -f "$OUTPUT_DIR/$batch_id/processing_summary.txt" ]; then
        echo "Skipping $batch_id (already processed)"
        continue
    fi
    
    echo "Processing $batch_id..."
    ./scripts/workflow.sh process $batch_id
    sleep 30
done
```

### Parallel Processing of Multiple States

```bash
# Process batches from different states in parallel
./scripts/workflow.sh process MD_2025-10-20 &
./scripts/workflow.sh process NC_2025-10-20 &
./scripts/workflow.sh process GA_2025-10-20 &

# Wait for all to complete
wait

echo "All batches processed"
```

## Monitoring

### Real-Time Job Monitoring

```bash
# Watch active jobs (updates every 10 seconds)
watch -n 10 'squeue -u $USER -A dash_agir --format="%.10i %.9P %.30j %.8T %.10M %.6D %.R"'

# Detailed job information
scontrol show job <JOB_ID>

# Job efficiency (after completion)
seff <JOB_ID>
```

### Log Monitoring

```bash
# Follow SLURM output in real-time
tail -f /project/dash_agir/matthew.kutugata/logs/snakemake_svs_pipeline_*.out

# Check for errors
grep -i error /project/dash_agir/matthew.kutugata/logs/snakemake_*.err

# Monitor specific batch processing
tail -f /project/dash_agir/matthew.kutugata/semifield-developed-images/$BATCH_ID/logs/*.log
```

### Database Queries

```bash
# Get batch summary
python scripts/db_manager.py --db /project/dash_agir/matthew.kutugata/pipeline_tracking.db summary

# Query specific batch
python scripts/db_manager.py --db /project/dash_agir/matthew.kutugata/pipeline_tracking.db \
    query "SELECT * FROM batches WHERE batch_id='MD_2025-10-22'"

# List all processed batches
python scripts/db_manager.py --db /project/dash_agir/matthew.kutugata/pipeline_tracking.db \
    query "SELECT batch_id, processing_status, updated_at FROM batches ORDER BY updated_at DESC"
```

### Progress Tracking

```bash
# Count processed images
BATCH_ID="MD_2025-10-22"
OUTPUT_DIR="/project/dash_agir/matthew.kutugata/semifield-developed-images/$BATCH_ID"

echo "RAW files: $(ls /90daydata/dash_agir/data/semifield-upload/$BATCH_ID/*.RAW | wc -l)"
echo "DNGs created: $(ls $OUTPUT_DIR/dngs/*.dng 2>/dev/null | wc -l)"
echo "JPGs created: $(ls $OUTPUT_DIR/images/*.jpg 2>/dev/null | wc -l)"
```

## Performance Tuning

### Optimize Resource Allocation

Edit `config/scinet.yaml`:

```yaml
slurm:
  max_parallel_jobs: 12  # Increase for faster processing (up to 20)
  mem_per_cpu: 4GB       # Increase if jobs fail with OOM
  cpus_per_task: 4       # Threads per image conversion

processing:
  threads_per_image: 4   # Match with cpus_per_task
```

### Benchmark Your Batch

```bash
# Time a small batch
time snakemake --config batch_id=<TEST_BATCH> --cores 4

# Check resource usage
sacct -j <JOB_ID> --format=JobID,JobName,MaxRSS,Elapsed,State
```

### Storage Optimization

```bash
# Clean up intermediate DNGs after JPG creation
sed -i 's/cleanup_dngs: false/cleanup_dngs: true/' config/scinet.yaml

# Move completed batches to long-term storage
rsync -av /project/dash_agir/matthew.kutugata/semifield-developed-images/$BATCH_ID/ \
    /project/dash_agir/semifield-developed-images/$BATCH_ID/
```

## Tips and Best Practices

1. **Always use compute nodes** for processing, never login nodes
2. **Stagger job submissions** (30-60 seconds between batches) to avoid overloading the scheduler
3. **Monitor storage quotas** on scratch (/90daydata has 90-day retention)
4. **Keep DNGs for archival** unless storage is constrained
5. **Use dry-run** (`--dry-run`) before large batch operations
6. **Check logs immediately** if a job fails to diagnose issues quickly
7. **Save processing summaries** for documentation and quality control

## Common Tasks

### Reprocess Failed Images

```bash
# Identify failed conversions
grep -r "ERROR" /project/dash_agir/matthew.kutugata/semifield-developed-images/$BATCH_ID/logs/

# Force rerun
snakemake --config batch_id=$BATCH_ID --forcerun convert_dng_to_jpg --cores 4
```

### Extract Processing Statistics

```bash
# From summary file
cat /project/dash_agir/matthew.kutugata/semifield-developed-images/$BATCH_ID/processing_summary.txt

# From SLURM stats
sacct -j <JOB_ID> --format=JobID,JobName,MaxRSS,Elapsed,CPUTime,State --parsable2
```

### Archive Completed Batches

```bash
# Create tarball
tar -czf $BATCH_ID.tar.gz \
    /project/dash_agir/matthew.kutugata/semifield-developed-images/$BATCH_ID/

# Move to archive location
mv $BATCH_ID.tar.gz /project/dash_agir/archive/
```

## Need Help?

- Review [Troubleshooting Guide](TROUBLESHOOTING.md)
- Check [Architecture Documentation](ARCHITECTURE.md)
- Contact: matthew.kutugata@usda.gov
