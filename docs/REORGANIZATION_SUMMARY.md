# Repository Reorganization Summary

**Date:** December 3, 2025
**Purpose:** Streamline svs-raw-api into a production-ready Snakemake pipeline for USDA SciNet HPC

## What Changed

### 🆕 New Files Created

#### Core Pipeline Files
- **`Snakefile`** - Main Snakemake workflow at repository root
- **`config/config.yaml`** - Unified configuration file
- **`config/environment.yaml`** - Conda environment specification
- **`config/cluster.yaml`** - SLURM cluster parameters
- **`config/slurm/config.yaml`** - Snakemake SLURM profile

#### Workflow Scripts
- **`workflow/scripts/raw_to_dng.py`** - RAW to DNG conversion script
- **`workflow/scripts/dng_to_jpg.py`** - DNG to JPG conversion script

#### SLURM Integration
- **`slurm/submit_snakemake.sh`** - Main SLURM submission script

#### User Scripts
- **`scripts/process_batch.sh`** - Quick batch processing wrapper
- **`scripts/setup_environment.sh`** - Environment setup helper
- **`scripts/list_batches.sh`** - List available batches

#### Documentation
- **`README.md`** - Comprehensive main documentation
- **`docs/QUICKSTART.md`** - Quick start guide
- **`docs/USAGE.md`** - Advanced usage patterns
- **`docs/REORGANIZATION_SUMMARY.md`** - This file

### 📦 Reorganized Files

#### Documentation
- Moved `docs/v1_snakemake_only/` → `docs/archive/v1_snakemake_only/`
- Moved `docs/v1_wo_snakemake/` → `docs/archive/v1_wo_snakemake/`
- Moved `docs/v2_full_int_guide_wo_py/` → `docs/archive/v2_full_int_guide_wo_py/`
- Created `docs/archive/README.md` - Archive index

#### Configuration
- Kept `conf/scinet.yaml` for backward compatibility
- Moved SVS tags to `data/profiles/svs_tags.yaml` (referenced in new config)

### 🔧 Modified Files

- **`pyproject.toml`** - Added `snakemake>=7.32.0` dependency
- **`conf/scinet.yaml`** - No changes (kept for compatibility)

### ⚠️ Deprecated Files (Not Removed)

The following files are kept for reference but are no longer the primary workflow:

- `slurm/process_batch.sh` - Old SLURM script (superseded by `submit_snakemake.sh`)
- `slurm/array_job.sh` - Old array job script
- `examples/*.py` - Example scripts (still useful for calibration)

## New Directory Structure

```
svs-raw-api/
├── Snakefile                      # ← NEW: Main workflow
├── README.md                      # ← UPDATED: Comprehensive docs
├── pyproject.toml                 # ← UPDATED: Added snakemake
│
├── config/                        # ← NEW: Unified config
│   ├── config.yaml               #    Main configuration
│   ├── environment.yaml          #    Conda environment
│   ├── cluster.yaml              #    SLURM parameters
│   └── slurm/
│       └── config.yaml           #    Snakemake profile
│
├── workflow/                      # ← NEW: Snakemake workflow
│   └── scripts/
│       ├── raw_to_dng.py         #    RAW conversion
│       └── dng_to_jpg.py         #    JPG conversion
│
├── slurm/
│   ├── submit_snakemake.sh       # ← NEW: Main submission
│   ├── process_batch.sh          #    (old, kept)
│   ├── array_job.sh              #    (old, kept)
│   └── setup_once.sh             #    (old, kept)
│
├── scripts/
│   ├── process_batch.sh          # ← NEW: Quick wrapper
│   ├── setup_environment.sh      # ← NEW: Setup helper
│   ├── list_batches.sh           # ← NEW: Batch lister
│   └── ...                       #    (other utilities)
│
├── docs/
│   ├── QUICKSTART.md             # ← NEW: Quick start
│   ├── USAGE.md                  # ← NEW: Advanced usage
│   ├── REORGANIZATION_SUMMARY.md # ← NEW: This file
│   └── archive/                  # ← NEW: Old docs
│       ├── v1_snakemake_only/
│       ├── v1_wo_snakemake/
│       ├── v2_full_int_guide_wo_py/
│       └── README.md
│
├── src/svs_raw_api/              # (unchanged)
├── data/profiles/                 # (unchanged)
├── examples/                      # (unchanged)
├── tests/                         # (unchanged)
└── conf/                          # (kept for compatibility)
    └── scinet.yaml
```

## Migration Guide

### For Existing Users

If you were using the old workflow:

**Old way:**
```bash
sbatch slurm/process_batch.sh MD_2025-10-22
```

**New way:**
```bash
./scripts/process_batch.sh MD_2025-10-22
# or
sbatch slurm/submit_snakemake.sh MD_2025-10-22
```

### Configuration Migration

**Old:** `conf/scinet.yaml` (still works!)

**New:** `config/config.yaml` (recommended)

You can continue using `conf/scinet.yaml` with the old scripts, or migrate to the new Snakemake workflow with `config/config.yaml`.

## Benefits of Reorganization

### ✨ Improvements

1. **Standardized Pipeline**
   - Industry-standard Snakemake workflow
   - Follows best practices for HPC pipelines

2. **Better Documentation**
   - Comprehensive README with quick start
   - Separate guides for different use cases
   - Archived old docs for reference

3. **Easier to Use**
   - Simple wrapper scripts: `./scripts/process_batch.sh`
   - Setup script: `./scripts/setup_environment.sh`
   - Batch listing: `./scripts/list_batches.sh`

4. **More Flexible**
   - Easy to customize via `config/config.yaml`
   - SLURM profile for different clusters
   - Modular workflow scripts

5. **Better Monitoring**
   - Snakemake progress tracking
   - Per-image logging
   - Batch summaries

6. **Fault Tolerance**
   - Automatic job retry
   - Resume from checkpoint
   - Smart dependency resolution

### 🚀 Performance

No change in performance - same parallel execution:
- 12 parallel SLURM jobs
- 4 cores × 16GB RAM per job
- ~20-30 minutes for 100 images

### 📊 Resource Usage

Slightly lower overhead:
- Snakemake main job: 2 cores, 8GB RAM
- Worker jobs: 4 cores, 16GB RAM each (same as before)

## Testing Checklist

Before using in production, verify:

- [ ] `config/config.yaml` paths are correct
- [ ] `scripts/setup_environment.sh` runs successfully
- [ ] Package installs: `pip install -e .`
- [ ] Snakemake available: `snakemake --version`
- [ ] RawTherapee path correct
- [ ] Test batch processes: `./scripts/process_batch.sh TEST`
- [ ] Output files created in expected locations
- [ ] Logs readable and informative

## Rollback Instructions

If you need to revert to the old workflow:

```bash
# Use old SLURM script directly
sbatch slurm/process_batch.sh MD_2025-10-22

# Or use old config
python -m svs_raw_api.cli \
    --config conf/scinet.yaml \
    --input /90daydata/.../MD_2025-10-22 \
    --output /project/.../MD_2025-10-22/images \
    --batch-id MD_2025-10-22
```

The old scripts are still present and functional.

## Future Enhancements

Potential additions:

1. **Database Integration**
   - Track processing status in SQLite/PostgreSQL
   - Integration with Globus for data transfer

2. **Web Dashboard**
   - Monitor processing status
   - View batch summaries
   - Download results

3. **Multi-State Support**
   - Automatic profile selection by state
   - Batch profile management

4. **Quality Control**
   - Automated image quality checks
   - Outlier detection
   - Exposure verification

## Support

For questions or issues:

- **Documentation:** `docs/` directory
- **Quick help:** `docs/QUICKSTART.md`
- **Advanced topics:** `docs/USAGE.md`
- **Issues:** Open GitHub issue
- **Contact:** matthew.kutugata@usda.gov

## Acknowledgments

This reorganization consolidates insights from:
- v1 three-tier pipeline documentation
- v1 Snakemake integration attempts
- v2 complete integration guide

All previous work has been preserved in `docs/archive/` for reference.
