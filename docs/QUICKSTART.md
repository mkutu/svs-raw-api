# Quick Start Guide

Get up and running with the SVS RAW processing pipeline in 5 minutes.

## Prerequisites

- Access to USDA SciNet Ceres HPC
- RAW image batch already on Ceres at `/90daydata/dash_agir/data/semifield-upload/`

## Step 1: Setup (One-Time)

```bash
# SSH to Ceres
ssh matthew.kutugata@ceres.scinet.usda.gov

# Clone repository
cd ~/repos
git clone <repo-url> svs-raw-api
cd svs-raw-api

# Install package
source /project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep/bin/activate
pip install -e . --no-deps
```

## Step 2: Configure Paths (One-Time)

Edit `config/config.yaml`:

```yaml
paths:
  repo_root: /home/YOUR_USERNAME/repos/svs-raw-api
  output_dir: /project/dash_agir/YOUR_USERNAME/semifield-developed-images
  rawtherapee_cli: /path/to/your/rawtherapee-cli
```

Find RawTherapee:
```bash
find ~ -name "rawtherapee-cli" 2>/dev/null
```

## Step 3: Process Your First Batch

```bash
# Process a batch
./scripts/process_batch.sh MD_2025-10-22

# Monitor progress
squeue -u $USER
```

That's it! The pipeline will:
1. Submit a Snakemake job to SLURM
2. Spawn 12 parallel processing jobs
3. Convert RAW → DNG → JPG
4. Save results to `/project/.../semifield-developed-images/MD_2025-10-22/`

## Step 4: Check Results

```bash
# View processing summary
cat /project/dash_agir/matthew.kutugata/semifield-developed-images/MD_2025-10-22/processing_summary.txt

# List output files
ls /project/dash_agir/matthew.kutugata/semifield-developed-images/MD_2025-10-22/images/
```

## Common Commands

```bash
# Check queue
squeue -u $USER

# View job output
tail -f /project/dash_agir/matthew.kutugata/logs/snakemake_*.out

# Process multiple batches
for batch in MD_2025-10-{22,23,24}; do
    ./scripts/process_batch.sh $batch
done
```

## Troubleshooting

### Batch not found
```bash
# Check if batch exists
ls /90daydata/dash_agir/data/semifield-upload/
```

### Package not found
```bash
# Reinstall
cd ~/repos/svs-raw-api
pip install -e . --no-deps
```

### Job fails
```bash
# Check error log
cat /project/dash_agir/matthew.kutugata/logs/snakemake_<job-id>.err

# Check individual image logs
ls /project/.../MD_2025-10-22/logs/
```

## Next Steps

- Read the [full README](../README.md) for detailed documentation
- See [USAGE.md](USAGE.md) for advanced features
- See [INSTALL.md](INSTALL.md) for detailed setup

---

**Need help?** Contact matthew.kutugata@usda.gov or open an issue on GitHub
