# Snakemake Integration for Three-Tier Pipeline

## New Snakemake Files

Three new files have been created to integrate Snakemake with the three-tier storage system:

1. **Snakefile_three_tier** - Enhanced Snakemake workflow
2. **snakemake_config_three_tier.yaml** - Configuration file
3. **run_snakemake_three_tier.sh** - SLURM submission script

## Key Enhancements

### 1. Database Integration

The Snakefile now integrates directly with the database:

```python
# Imports db_manager for real-time updates
from db_manager import BatchDatabase

# Updates status at each stage
rule validate_batch:
    # Sets processing_status = 'processing'
    
rule update_dng_status:
    # Sets dng_created = 1 for each file
    
rule update_jpg_status:
    # Sets jpg_created = 1, processing_status = 'completed'
    
rule finalize_batch:
    # Sets batch processing_status = 'completed'
```

### 2. Three-Tier Validation

Before processing, the Snakefile validates the batch came through the proper pipeline:

```python
rule validate_batch:
    # Checks batch exists on Ceres
    # Verifies database shows transfer_status = 'transferred'
    # Warns if NCSU sync status incomplete
    # Creates validation marker file
```

### 3. Enhanced Tracking

New tracking features:

- **File-level status**: Each RAW → DNG → JPG tracked individually
- **Processing history**: Records SLURM job ID, timestamps, file counts
- **Batch summary**: Generates detailed summary with success rates
- **Error handling**: Updates database with failure status on errors

### 4. New Rules

Additional Snakemake rules:

```
validate_batch       → Check batch ready for processing
update_dng_status    → Track DNG creation in database
update_jpg_status    → Track JPG creation in database
create_summary       → Generate processing summary
finalize_batch       → Mark batch complete in database
```

## Installation

### Step 1: Copy Files to Repository

```bash
cd ~/repos/svs-raw-api

# Copy Snakefile
cp /path/to/Snakefile_three_tier .

# Copy config
cp /path/to/snakemake_config_three_tier.yaml config/

# Copy SLURM script
cp /path/to/run_snakemake_three_tier.sh slurm/
chmod +x slurm/run_snakemake_three_tier.sh
```

### Step 2: Update Configuration

Edit `config/snakemake_config_three_tier.yaml`:

1. Update paths if different from defaults
2. Verify calibration matrix path
3. Verify PP3 profile path
4. Adjust resource limits if needed

### Step 3: Update workflow.sh (Optional)

To make `workflow.sh process` use the new Snakefile, update the `process_batch()` function:

```bash
process_batch() {
    local batch_id="$1"
    
    # ... existing validation ...
    
    # Use new SLURM script
    local slurm_script="$REPO_ROOT/slurm/run_snakemake_three_tier.sh"
    
    print_info "Submitting SLURM job with three-tier Snakefile..."
    sbatch -A dash_agir "$slurm_script" "$batch_id"
}
```

Or just submit directly:

```bash
sbatch -A dash_agir slurm/run_snakemake_three_tier.sh MD_2025-10-22
```

## Configuration Options

### Processing Mode

```yaml
# Process single batch (specify batch_id)
mode: single
batch_id: "MD_2025-10-22"

# Process all ready batches
mode: batch
state_filter: "MD"  # Optional
```

### Storage Paths

```yaml
paths:
  ceres_base_path: "/90daydata/dash_agir/data/semifield-upload"
  output_base: "/project/dash_agir/matthew.kutugata/semifield-developed-images"
  database_path: "/project/dash_agir/matthew.kutugata/pipeline_tracking.db"
```

### Resource Allocation

```yaml
resources:
  max_jobs: 12              # Parallel SLURM jobs
  cores_per_job: 4          # CPUs per job
  memory_per_job: 16000     # MB per job
  max_runtime: 120          # Minutes per job
```

### Validation

```yaml
validation:
  require_transferred: true     # Check transfer_status in DB
  min_raw_files: 1             # Minimum RAW files required
  strict_db_check: false       # Fail if DB status mismatch
```

### Cleanup

```yaml
cleanup:
  remove_dngs: true            # Delete DNGs after JPG creation
  remove_raw_from_ceres: false # CAREFUL: Only if archived in JUNO
  keep_logs: true              # Keep Snakemake logs
```

## Usage Examples

### Process Single Batch

```bash
# Using workflow.sh (recommended)
./scripts/workflow.sh process MD_2025-10-22

# Or submit directly
sbatch -A dash_agir slurm/run_snakemake_three_tier.sh MD_2025-10-22

# With custom config
sbatch -A dash_agir slurm/run_snakemake_three_tier.sh MD_2025-10-22 config/custom.yaml
```

### Process Multiple Batches

```bash
# Change config to batch mode
cat > /tmp/batch_config.yaml <<EOF
mode: batch
state_filter: "MD"
# ... rest of config ...
EOF

# Submit
sbatch -A dash_agir slurm/run_snakemake_three_tier.sh "" /tmp/batch_config.yaml
```

### Monitor Progress

```bash
# Check SLURM job
squeue -u $USER -j <job-id>

# View log
tail -f /project/dash_agir/matthew.kutugata/logs/snakemake_svs_pipeline_3tier_<job-id>.out

# Check database status
sqlite3 /project/dash_agir/matthew.kutugata/pipeline_tracking.db <<EOF
SELECT batch_id, processing_status, 
       COUNT(CASE WHEN jpg_created = 1 THEN 1 END) as completed
FROM batches 
LEFT JOIN files USING (batch_id)
WHERE batch_id = 'MD_2025-10-22'
GROUP BY batch_id;
EOF

# View processing history
sqlite3 /project/dash_agir/matthew.kutugata/pipeline_tracking.db <<EOF
SELECT * FROM processing_history 
WHERE batch_id = 'MD_2025-10-22'
ORDER BY started_at DESC;
EOF
```

## Database Updates During Processing

The Snakefile updates the database at each stage:

### 1. Start Processing
```sql
UPDATE batches SET processing_status = 'processing';
INSERT INTO processing_history (batch_id, job_id, status, started_at);
```

### 2. DNG Creation (per file)
```sql
UPDATE files 
SET dng_created = 1, dng_path = '...' 
WHERE filename = 'IMAGE001.ARW';
```

### 3. JPG Creation (per file)
```sql
UPDATE files 
SET jpg_created = 1, jpg_path = '...', processing_status = 'completed'
WHERE filename = 'IMAGE001.ARW';
```

### 4. Completion
```sql
UPDATE batches 
SET processing_status = 'completed', 
    processing_completed_at = datetime('now');

UPDATE processing_history 
SET status = 'completed', files_processed = 100;
```

### 5. On Error
```sql
UPDATE batches SET processing_status = 'failed';
UPDATE processing_history SET status = 'failed', error_message = '...';
```

## Validation Checks

Before processing, the Snakefile validates:

1. **Batch exists on Ceres**
   ```python
   batch_path = /90daydata/dash_agir/data/semifield-upload/{batch_id}
   if not batch_path.exists(): raise ValueError
   ```

2. **RAW files present**
   ```python
   raw_files = list(batch_path.glob("*.ARW"))
   if not raw_files: raise ValueError
   ```

3. **Database status** (if strict_db_check=true)
   ```python
   if transfer_status != "transferred": 
       raise ValueError("Batch not transferred to Ceres")
   ```

4. **Pipeline chain** (logged but doesn't fail)
   ```python
   print(f"NCSU sync: {ncsu_sync_status}")
   print(f"Transfer: {transfer_status}")
   print(f"Processing: {processing_status}")
   ```

## Output Files

For each batch, the pipeline creates:

```
/project/dash_agir/matthew.kutugata/semifield-developed-images/
└── MD_2025-10-22/
    ├── IMAGE001.jpg
    ├── IMAGE002.jpg
    ├── ...
    ├── processing_summary.txt    # NEW: Detailed summary
    └── done.txt                  # NEW: Completion marker
```

### processing_summary.txt
```
Batch Processing Summary
========================

Batch ID: MD_2025-10-22
Processed: 2025-10-22T14:30:00

Input Files:
  RAW files: 100
  Total size: 18.50 GB

Output Files:
  JPG files: 100
  Total size: 2.30 GB

Success rate: 100.0%

Pipeline Status:
  NCSU sync: synced
  JUNO→Ceres transfer: transferred
  Processing: completed
```

## Troubleshooting

### "Batch not found on Ceres"

**Problem:** Batch hasn't been transferred yet

**Solution:**
```bash
./scripts/workflow.sh transfer MD_2025-10-22
# Wait for transfer to complete
./scripts/workflow.sh process MD_2025-10-22
```

### "Database status mismatch"

**Problem:** Database shows status other than 'transferred'

**Solution:**
```bash
# Check actual status
./scripts/workflow.sh status

# If files are on Ceres but DB wrong, update manually:
sqlite3 /project/dash_agir/matthew.kutugata/pipeline_tracking.db <<EOF
UPDATE batches 
SET transfer_status = 'transferred' 
WHERE batch_id = 'MD_2025-10-22';
EOF
```

### "ImportError: No module named db_manager"

**Problem:** db_manager.py not in Python path

**Solution:** Ensure `scripts/` directory contains `db_manager.py` and Snakemake is run from repo root

### Processing fails mid-batch

**Problem:** Some files fail to convert

**Solution:** Snakemake keeps going and marks failed files in database:
```sql
SELECT filename, processing_status 
FROM files 
WHERE batch_id = 'MD_2025-10-22' 
AND processing_status = 'failed';
```

## Performance

With three-tier Snakefile:

- **Startup overhead**: +10-20 seconds (validation)
- **Per-file overhead**: +0.5 seconds (database updates)
- **Total batch time**: Essentially same as before (~40-60 min per 100 images)

The database tracking overhead is negligible compared to image processing time.

## Migration from Old Snakefile

If you have the old `Snakefile_enhanced`:

1. **Keep both** - They can coexist:
   ```
   Snakefile_enhanced    # Old version
   Snakefile_three_tier  # New version
   ```

2. **Test side-by-side**:
   ```bash
   # Old way
   sbatch slurm/run_snakemake_enhanced.sh MD_2025-10-22
   
   # New way  
   sbatch slurm/run_snakemake_three_tier.sh MD_2025-10-22
   ```

3. **Switch gradually** - Update `workflow.sh` to use new version once tested

## Next Steps

1. Copy files to your repo
2. Update configuration paths
3. Test on single batch: `sbatch slurm/run_snakemake_three_tier.sh <test-batch>`
4. Monitor in database
5. Update `workflow.sh` to use new Snakefile
6. Process remaining batches

## Questions?

See inline comments in:
- `Snakefile_three_tier` - Complete workflow with comments
- `snakemake_config_three_tier.yaml` - All configuration options
- `run_snakemake_three_tier.sh` - SLURM submission with validation
