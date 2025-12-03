# SVS RAW Processing Pipeline - Complete Integration Package

## 📦 What's Included

This package integrates your `svs_raw_api` image processing code with the three-tier pipeline infrastructure, providing a complete automated system for processing Sony ARW/RAW images from field cameras.

## 🎯 Complete Data Flow

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  NCSU NFS   │  →   │ JUNO Archive│  →   │ Ceres /90day│  →   │   Process   │  →   │JUNO Archive │
│  (Upload)   │      │ (Long-term) │      │  (Scratch)  │      │ RAW→DNG→JPG │      │  (Results)  │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
     Primary           Permanent             Temporary            Parallel              Final
     Storage           Archive               Processing           Compute               Storage
```

## 📄 New Files Created (4 Total)

All files are in `/mnt/user-data/outputs/`:

### 1. Core Processing Files

| File | Purpose | Install To |
|------|---------|------------|
| **Snakefile** | Snakemake workflow for parallel RAW→DNG→JPG conversion | `~/repos/svs-raw-api/Snakefile` |
| **snakemake_config.yaml** | Configuration for processing pipeline | `~/repos/svs-raw-api/config/snakemake_config.yaml` |
| **run_snakemake.sh** | SLURM script to submit processing jobs | `~/repos/svs-raw-api/slurm/run_snakemake.sh` |
| **workflow_integrated.sh** | Updated workflow manager with processing integration | `~/repos/svs-raw-api/scripts/workflow.sh` |

### 2. Documentation & Tools

| File | Purpose |
|------|---------|
| **PROCESSING_INTEGRATION_GUIDE.md** | Complete setup and usage guide |
| **test_setup.sh** | Verification script to test your setup |

## 🚀 Quick Start

### Step 1: Download Files

```bash
# On your local machine, download all files from the outputs directory
# Then transfer to Ceres:

scp Snakefile matthew.kutugata@ceres-dtn:~/repos/svs-raw-api/
scp snakemake_config.yaml matthew.kutugata@ceres-dtn:~/repos/svs-raw-api/config/
scp run_snakemake.sh matthew.kutugata@ceres-dtn:~/repos/svs-raw-api/slurm/
scp workflow_integrated.sh matthew.kutugata@ceres-dtn:~/repos/svs-raw-api/scripts/workflow.sh
scp test_setup.sh matthew.kutugata@ceres-dtn:~/repos/svs-raw-api/
scp PROCESSING_INTEGRATION_GUIDE.md matthew.kutugata@ceres-dtn:~/repos/svs-raw-api/docs/
```

### Step 2: Make Scripts Executable

```bash
# On Ceres
cd ~/repos/svs-raw-api
chmod +x slurm/run_snakemake.sh
chmod +x scripts/workflow.sh
chmod +x test_setup.sh
```

### Step 3: Install Dependencies

```bash
# Get compute node
salloc -A dash_agir -t 01:00:00 -c 4

# Activate conda environment
source /project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep/bin/activate

# Install svs_raw_api
cd ~/repos/svs-raw-api
pip install -e . --no-deps

# Install Snakemake
pip install snakemake --break-system-packages

# Exit compute node
exit
```

### Step 4: Verify Setup

```bash
cd ~/repos/svs-raw-api
./test_setup.sh
```

### Step 5: Process Your First Batch

```bash
# Complete pipeline (sync → transfer → process)
./scripts/workflow.sh full-pipeline MD_2025-10-22

# OR step by step:
./scripts/workflow.sh sync MD_2025-10-22      # NCSU → JUNO
./scripts/workflow.sh transfer MD_2025-10-22  # JUNO → Ceres
./scripts/workflow.sh process MD_2025-10-22   # RAW → DNG → JPG
```

## 📊 What This System Does

### Automated Parallel Processing

**Before (Manual):**
```bash
# Process each image one at a time
for raw in *.ARW; do
    python convert_raw_to_dng.py $raw
    rawtherapee-cli -c $raw.dng
done
# Time: ~3 minutes per image = 5 hours for 100 images
```

**After (Automated):**
```bash
./scripts/workflow.sh process MD_2025-10-22
# Snakemake automatically:
# - Discovers all RAW files
# - Spawns 12 parallel SLURM jobs
# - Processes RAW → DNG → JPG
# - Tracks progress and handles failures
# Time: ~20-30 minutes for 100 images
```

### Resource Management

**Per Batch Processing:**
- **12 parallel jobs** (can process 12 images simultaneously)
- **4 cores × 16GB RAM** per job
- **Total: 48 cores, 192GB RAM**
- **Automatic queue management** via SLURM
- **Checkpoint/resume** capability

### Workflow Integration

```bash
# Single command for complete pipeline
./scripts/workflow.sh full-pipeline MD_2025-10-22

# Automatically:
# 1. Checks if batch needs NCSU → JUNO sync
# 2. Transfers JUNO → Ceres scratch
# 3. Submits SLURM job for parallel processing
# 4. Tracks status in database
# 5. Archives results to JUNO
```

## 🎨 Processing Pipeline Details

### Stage 1: RAW → DNG (Adobe Format)

**What it does:**
- Reads Sony ARW raw sensor data
- Applies custom color calibration matrix
- Embeds metadata (camera tags, EXIF)
- Creates standardized DNG format

**Why DNG?**
- Industry-standard archival format
- Preserves full sensor data
- Compatible with Adobe tools
- Embeds custom calibrations

### Stage 2: DNG → JPG (Display Format)

**What it does:**
- Applies PP3 processing profile (RawTherapee)
- Demosaicing and color processing
- Sharpening and noise reduction
- Creates high-quality JPG for analysis

**Custom Profiles:**
- MD-specific color profile
- Optimized for SVS camera sensors
- Consistent across all images

## 📁 Output Structure

```
/project/dash_agir/matthew.kutugata/semifield-developed-images/
└── MD_2025-10-22/
    ├── dngs/                    # Archival DNGs (kept permanently)
    │   ├── MD_1760033880.dng
    │   ├── MD_1760033890.dng
    │   └── ...
    ├── images/                  # Final JPGs (analysis-ready)
    │   ├── MD_1760033880.jpg
    │   ├── MD_1760033890.jpg
    │   └── ...
    ├── logs/                    # Per-image processing logs
    │   ├── raw_to_dng_MD_1760033880.log
    │   ├── dng_to_jpg_MD_1760033880.log
    │   └── ...
    └── processing_summary.txt   # Batch processing summary
```

## ⚡ Performance

**Timing (100-image batch):**
```
NCSU → JUNO sync:        10-30 minutes (one-time)
JUNO → Ceres transfer:    5-10 minutes
RAW → DNG → JPG:         20-30 minutes (parallel)
─────────────────────────────────────────
Total first processing:   40-60 minutes
```

**Comparison:**
- **Sequential processing:** ~5 hours (3 min/image × 100)
- **Parallel processing:** ~25 minutes (12 images at once)
- **Speedup:** ~12x faster

## 🔧 Configuration

### Edit Processing Settings

`config/snakemake_config.yaml`:

```yaml
processing:
  height: 3024              # SVS camera height
  width: 4032               # SVS camera width
  threads_per_image: 4      # RawTherapee threads
  cleanup_dngs: false       # Keep DNGs after JPG creation
  cleanup_raw: false        # Keep RAW in scratch

slurm:
  max_parallel_jobs: 12     # Parallel processing jobs
  mem_per_cpu: 4GB          # Memory per core
  time: "04:00:00"          # Max job time
```

### Custom Color Profiles

For different states or lighting conditions:

```bash
# Create state-specific profile
cp data/profiles/MD_calibration_matrix_optimized.npy \
   data/profiles/NC_calibration_matrix_optimized.npy

# Create state-specific PP3
cp data/profiles/MD_shr661_raw16.pp3 \
   data/profiles/NC_custom.pp3

# Edit NC_custom.pp3 with your settings
```

## 📋 Common Workflows

### Daily Processing

```bash
# Check for new batches
./scripts/workflow.sh check-missing

# Process each new batch
./scripts/workflow.sh full-pipeline MD_2025-10-22
./scripts/workflow.sh full-pipeline MD_2025-10-23
```

### Batch Processing Multiple Batches

```bash
# Get list of batches needing processing
batches=$(sqlite3 /project/dash_agir/matthew.kutugata/pipeline_tracking.db \
    "SELECT batch_id FROM batches WHERE processing_status='pending' LIMIT 5")

# Process all
for batch in $batches; do
    ./scripts/workflow.sh process $batch
    sleep 30  # Stagger submissions
done

# Monitor
watch -n 30 'squeue -u $USER -A dash_agir'
```

### Reprocess Failed Images

```bash
# Snakemake automatically detects missing outputs and reprocesses
./scripts/workflow.sh process MD_2025-10-22

# Or manually delete outputs and reprocess
rm /project/dash_agir/matthew.kutugata/semifield-developed-images/MD_2025-10-22/images/*.jpg
./scripts/workflow.sh process MD_2025-10-22
```

## 🔍 Monitoring

### Check Processing Status

```bash
# Overall pipeline status
./scripts/workflow.sh status

# Check specific batch
sqlite3 /project/dash_agir/matthew.kutugata/pipeline_tracking.db \
    "SELECT * FROM batches WHERE batch_id='MD_2025-10-22'"

# Monitor SLURM jobs
squeue -u $USER -A dash_agir
watch -n 10 'squeue -u $USER -A dash_agir'
```

### View Logs

```bash
# SLURM job output
tail -f /project/dash_agir/matthew.kutugata/logs/snakemake_svs_raw_process_<job-id>.out

# Snakemake log
tail -f /project/dash_agir/matthew.kutugata/logs/snakemake_<batch-id>_<job-id>.log

# Per-image logs
tail /project/dash_agir/matthew.kutugata/semifield-developed-images/<batch-id>/logs/*.log
```

### Check Outputs

```bash
# Count processed images
ls -1 /project/dash_agir/matthew.kutugata/semifield-developed-images/MD_2025-10-22/images/*.jpg | wc -l

# View processing summary
cat /project/dash_agir/matthew.kutugata/semifield-developed-images/MD_2025-10-22/processing_summary.txt

# Check file sizes
du -sh /project/dash_agir/matthew.kutugata/semifield-developed-images/MD_2025-10-22/*
```

## 🐛 Troubleshooting

### "svs_raw_api not found"

```bash
cd ~/repos/svs-raw-api
source /project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep/bin/activate
pip install -e . --no-deps
```

### "RawTherapee CLI not found"

Update path in `config/snakemake_config.yaml`:
```yaml
paths:
  rawtherapee_cli: /path/to/your/rawtherapee-cli
```

### Processing job fails

```bash
# Check SLURM error log
cat /project/dash_agir/matthew.kutugata/logs/snakemake_svs_raw_process_<job-id>.err

# Check individual image logs
grep -i error /project/dash_agir/matthew.kutugata/semifield-developed-images/<batch-id>/logs/*.log
```

### Out of memory

Reduce parallel jobs in `config/snakemake_config.yaml`:
```yaml
slurm:
  max_parallel_jobs: 8  # Reduce from 12
```

## 📚 Documentation

- **Setup Guide:** `PROCESSING_INTEGRATION_GUIDE.md` - Complete installation and configuration
- **Pipeline Guide:** `THREE_TIER_SETUP.md` - Three-tier infrastructure overview
- **Quick Reference:** `THREE_TIER_QUICK_REF.md` - Common commands
- **Architecture:** `ARCHITECTURE_DIAGRAM.txt` - System architecture

## ✅ Verification Checklist

Run the verification script:
```bash
cd ~/repos/svs-raw-api
./test_setup.sh
```

Manual checks:
- [ ] Snakemake installed
- [ ] svs_raw_api package installed
- [ ] RawTherapee CLI accessible
- [ ] Configuration files in place
- [ ] Profile files (svs_tags.yaml, color matrix, PP3) present
- [ ] SLURM access working
- [ ] Globus logged in
- [ ] Test batch processed successfully

## 🎓 What You've Gained

**Automation:**
- ✅ No manual image conversion
- ✅ Parallel processing (12x speedup)
- ✅ Automatic error handling
- ✅ Progress tracking

**Reliability:**
- ✅ Database tracking
- ✅ Checkpoint/resume capability
- ✅ Consistent processing profiles
- ✅ Comprehensive logging

**Scalability:**
- ✅ Process hundreds of images
- ✅ Multiple batches simultaneously
- ✅ Resource management via SLURM
- ✅ Efficient storage use

## 🔗 Integration Points

This package integrates with your existing:
- `svs_raw_api` Python package (RAW conversion)
- RawTherapee CLI (DNG to JPG)
- Three-tier pipeline (data management)
- Database tracking (status monitoring)
- Globus transfers (data movement)

## 📞 Support

**Issues with:**
- **Setup:** See `PROCESSING_INTEGRATION_GUIDE.md`
- **Pipeline:** See `THREE_TIER_SETUP.md`
- **Commands:** See `THREE_TIER_QUICK_REF.md`
- **Snakemake:** https://snakemake.readthedocs.io/
- **SLURM:** https://slurm.schedmd.com/documentation.html
- **SCINet:** scinet_vrsc@usda.gov

## 🚦 You're Ready!

1. ✅ Download and install files
2. ✅ Run test_setup.sh
3. ✅ Process test batch
4. ✅ Start processing your data!

```bash
./scripts/workflow.sh full-pipeline MD_2025-10-22
```

Happy processing! 🎉
