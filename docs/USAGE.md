# Usage Guide

Advanced usage patterns for the SVS RAW processing pipeline.

## Processing Workflows

### Single Batch Processing

```bash
# Method 1: Quick script (recommended)
./scripts/process_batch.sh MD_2025-10-22

# Method 2: Direct SLURM submission
sbatch slurm/submit_snakemake.sh MD_2025-10-22

# Method 3: Manual Snakemake (for debugging)
cd ~/repos/svs-raw-api
snakemake --profile config/slurm --config batch_id=MD_2025-10-22 --verbose
```

### Batch Processing

```bash
# Process multiple batches sequentially
for batch in MD_2025-10-{22,23,24,25}; do
    echo "Submitting $batch"
    ./scripts/process_batch.sh $batch
    sleep 10  # Stagger submissions to avoid overloading scheduler
done
```

### Parallel Batch Processing

```bash
# Submit multiple batches in parallel (careful with resource limits!)
parallel -j 3 "./scripts/process_batch.sh {}" ::: MD_2025-10-{22..30}
```

### Reprocessing Failed Images

Snakemake automatically detects missing outputs and reprocesses only what's needed:

```bash
# Just resubmit - Snakemake will skip completed files
./scripts/process_batch.sh MD_2025-10-22
```

Force complete reprocessing:

```bash
# Delete outputs
rm -rf /project/dash_agir/matthew.kutugata/semifield-developed-images/MD_2025-10-22

# Reprocess
./scripts/process_batch.sh MD_2025-10-22
```

## Configuration

### Per-Batch Configuration

Override configuration for specific batches:

```bash
# Create batch-specific config
cp config/config.yaml config/config_NC.yaml

# Edit for North Carolina batches
nano config/config_NC.yaml

# Use custom config
snakemake \
    --configfile config/config_NC.yaml \
    --profile config/slurm \
    --config batch_id=NC_2025-10-22
```

### Custom Color Profiles

#### For a Different State

```bash
# 1. Create calibration matrix for new state
python examples/create_calibration_profile.py \
    --input /path/to/colorchecker/NC_calibration.RAW \
    --output data/profiles/NC_calibration_matrix_optimized.npy

# 2. Copy and customize PP3 profile
cp data/profiles/MD_shr661_raw16.pp3 data/profiles/NC_custom.pp3

# 3. Update config.yaml
nano config/config.yaml
# Change:
#   color_matrix: data/profiles/NC_calibration_matrix_optimized.npy
#   pp3_profile: data/profiles/NC_custom.pp3
```

#### For Different Lighting Conditions

```bash
# Create profile variants
cp data/profiles/MD_shr661_raw16.pp3 data/profiles/MD_cloudy.pp3
cp data/profiles/MD_shr661_raw16.pp3 data/profiles/MD_sunny.pp3

# Edit in RawTherapee GUI or text editor
rawtherapee data/profiles/MD_cloudy.pp3
```

### Resource Tuning

Edit `config/config.yaml`:

```yaml
slurm:
  # Increase parallelism (more jobs, but each smaller)
  max_parallel_jobs: 16
  cpus_per_job: 3
  mem_per_job_mb: 12000

  # Or decrease for stability
  max_parallel_jobs: 8
  cpus_per_job: 6
  mem_per_job_mb: 20000
```

## Monitoring and Debugging

### Real-Time Monitoring

```bash
# Watch queue
watch -n 10 'squeue -u $USER -A dash_agir'

# Follow main log
tail -f /project/dash_agir/matthew.kutugata/logs/snakemake_<job-id>.out

# Monitor resource usage
sacct -j <job-id> --format=JobID,JobName,State,MaxRSS,Elapsed
```

### Detailed Debugging

```bash
# Dry run to see what will execute
snakemake --config batch_id=MD_2025-10-22 --dry-run --printshellcmds

# Check Snakemake DAG
snakemake --config batch_id=MD_2025-10-22 --dag | dot -Tpng > dag.png

# Verbose output
snakemake --config batch_id=MD_2025-10-22 --verbose --printshellcmds --reason
```

### Log Analysis

```bash
# Find failed jobs
grep -i "error\|failed" /project/.../logs/snakemake_*.out

# Check individual image logs
cd /project/.../MD_2025-10-22/logs/
grep -l "ERROR" *.log

# View errors from specific log
cat raw_to_dng_MD_1760033880.log
```

## Advanced Snakemake

### Cleaning Up

```bash
# Remove only intermediate files (keep final JPGs)
snakemake --config batch_id=MD_2025-10-22 cleanup

# Clean everything for a batch
snakemake --config batch_id=MD_2025-10-22 --delete-all-output
```

### Checkpoints and Resume

Snakemake automatically resumes from where it left off:

```bash
# Start processing
./scripts/process_batch.sh MD_2025-10-22

# Job fails midway...

# Just resubmit - only missing outputs will be processed
./scripts/process_batch.sh MD_2025-10-22
```

### Cluster Status

```bash
# Show cluster configuration
snakemake --config batch_id=MD_2025-10-22 --cluster-status

# Detailed cluster stats
snakemake --config batch_id=MD_2025-10-22 --detailed-summary
```

## Custom Workflows

### Processing Subset of Images

Edit Snakefile temporarily:

```python
# Add filter after RAW_FILES definition
RAW_FILES = [f for f in RAW_FILES if "morning" in f.name]
```

### Running Individual Rules

```bash
# Only convert RAW to DNG (no JPG)
snakemake \
    --config batch_id=MD_2025-10-22 \
    --profile config/slurm \
    --until raw_to_dng

# Only convert specific image
snakemake \
    --config batch_id=MD_2025-10-22 \
    /project/.../MD_2025-10-22/images/MD_1760033880.jpg
```

### Testing on Small Subset

```bash
# Create test batch with fewer images
mkdir -p /90daydata/dash_agir/data/semifield-upload/TEST
cp /90daydata/dash_agir/data/semifield-upload/MD_2025-10-22/*.RAW \
   /90daydata/dash_agir/data/semifield-upload/TEST/ | head -5

# Process test batch
./scripts/process_batch.sh TEST
```

## Performance Optimization

### Using /tmp for I/O

The pipeline automatically uses fast local storage (`/tmp`) during processing. This is configured in the SLURM submission script.

### Optimal Resource Allocation

```yaml
# For memory-intensive batches (high-resolution)
slurm:
  cpus_per_job: 4
  mem_per_job_mb: 20000

# For CPU-intensive batches (many images)
slurm:
  max_parallel_jobs: 16
  cpus_per_job: 3
```

### Benchmark Mode

```bash
# Run with benchmarking
snakemake \
    --config batch_id=MD_2025-10-22 \
    --profile config/slurm \
    --benchmark-extended
```

## Storage Management

### Disk Usage

```bash
# Check batch sizes
du -sh /project/.../semifield-developed-images/*

# Breakdown by file type
du -sh /project/.../MD_2025-10-22/dngs
du -sh /project/.../MD_2025-10-22/images
```

### Cleanup Strategies

```yaml
# Option 1: Keep DNGs (default)
processing:
  keep_dngs: true

# Option 2: Remove DNGs after JPG creation
processing:
  keep_dngs: false
```

Manual cleanup:

```bash
# Remove DNGs for specific batch (keep JPGs)
rm -rf /project/.../MD_2025-10-22/dngs/

# Archive old batches to tape storage
tar -czf MD_2025-10-22.tar.gz /project/.../MD_2025-10-22
# (then move to JUNO LTS)
```

## Integration with Other Tools

### Exporting File Lists

```bash
# List all processed JPGs
find /project/.../semifield-developed-images -name "*.jpg" > processed_images.txt

# Create CSV manifest
echo "batch_id,filename,size_mb,date" > manifest.csv
find /project/.../semifield-developed-images -name "*.jpg" -exec \
    bash -c 'echo "$(dirname {}),$(basename {}),$(du -m {} | cut -f1),$(date -r {} +%Y-%m-%d)"' \; \
    >> manifest.csv
```

### Batch Metadata

```bash
# Extract processing summaries
for batch in /project/.../semifield-developed-images/*; do
    cat $batch/processing_summary.txt >> all_batches_summary.txt
    echo "---" >> all_batches_summary.txt
done
```

## Troubleshooting Tips

### Common Issues

**"Permission denied" errors:**
```bash
# Check file permissions
ls -la /project/.../semifield-developed-images/
chmod -R u+rw /project/.../semifield-developed-images/
```

**"Out of space" on /tmp:**
```bash
# Check /tmp usage during job
df -h /tmp

# Reduce parallel jobs to limit /tmp usage
```

**Slow processing:**
```bash
# Check node load
scontrol show node <node-name>

# Use different partition
# Edit config/cluster.yaml: partition: medium
```

## Getting Help

- **Documentation:** `~/repos/svs-raw-api/docs/`
- **Snakemake docs:** https://snakemake.readthedocs.io/
- **SLURM guide:** https://scinet.usda.gov/guide/ceres/
- **Contact:** matthew.kutugata@usda.gov
