# Three-Tier Pipeline Quick Reference

## Storage Architecture

```
NCSU NFS â”€â”€syncâ”€â”€> JUNO LTS â”€â”€transferâ”€â”€> Ceres /90daydata â”€â”€processâ”€â”€> Final Output
(Primary)          (Archive)               (Scratch)                    (Project Dir)
```

## Essential Commands

### Initial Setup (One Time)

```bash
# 1. Find your NCSU endpoint
./scripts/find_ncsu_endpoint.sh

# 2. Update globus_manager.py with your NCSU endpoint and path

# 3. Initialize database
./scripts/workflow.sh init

# 4. Verify setup
./scripts/workflow.sh check-missing
```

### Daily Operations

```bash
# Check what's new in NCSU but not in JUNO
./scripts/workflow.sh check-missing
./scripts/workflow.sh check-missing MD    # Filter by state

# Sync single batch from NCSU to JUNO
./scripts/workflow.sh sync MD_2025-10-22

# Sync all missing batches
./scripts/workflow.sh sync-all

# Run complete pipeline for one batch
./scripts/workflow.sh full-pipeline MD_2025-10-22

# Check overall status
./scripts/workflow.sh status

# Interactive menu
./scripts/workflow.sh interactive
```

### Manual Step-by-Step

```bash
# Step 1: NCSU â†’ JUNO (if needed)
./scripts/workflow.sh sync MD_2025-10-22

# Step 2: JUNO â†’ Ceres
./scripts/workflow.sh transfer MD_2025-10-22

# Step 3: Process on Ceres
./scripts/workflow.sh process MD_2025-10-22
```

### Monitoring

```bash
# Check Globus transfer status
./scripts/workflow.sh check-task <task-id>
globus task show <task-id>
globus task list --filter-status=ACTIVE

# Check SLURM jobs
squeue -u $USER -A dash_agir
sacct -X -u $USER --starttime=today --format=JobID,JobName,State,Elapsed

# View pipeline status
./scripts/workflow.sh status

# Database queries
python scripts/db_manager.py --db /project/dash_agir/matthew.kutugata/pipeline_tracking.db summary
```

### Database Queries

```bash
# Connect to database
sqlite3 /project/dash_agir/matthew.kutugata/pipeline_tracking.db

# Useful queries (in SQLite shell):

# Batches needing NCSU sync
SELECT batch_id, ncsu_sync_status, date 
FROM batches 
WHERE ncsu_sync_status IN ('needed', 'unknown')
ORDER BY date DESC;

# Batches ready to transfer
SELECT batch_id, ncsu_sync_status, transfer_status 
FROM batches 
WHERE ncsu_sync_status = 'synced' 
AND transfer_status = 'pending'
ORDER BY date DESC;

# Full pipeline status
SELECT 
    batch_id,
    ncsu_sync_status,
    transfer_status,
    processing_status,
    date
FROM batches
ORDER BY date DESC
LIMIT 20;

# Recent NCSU syncs
SELECT 
    batch_id,
    status,
    bytes_transferred,
    started_at,
    completed_at
FROM ncsu_sync_history
ORDER BY started_at DESC
LIMIT 10;
```

## Batch Processing Multiple Batches

### Sync Multiple Batches

```bash
# Get list of missing batches
missing=$(./scripts/workflow.sh check-missing | grep "â€¢ " | awk '{print $2}')

# Sync each one
for batch in $missing; do
    echo "Syncing $batch..."
    ./scripts/workflow.sh sync $batch
    sleep 5  # Stagger submissions
done

# Monitor all transfers
watch -n 30 'globus task list --filter-status=ACTIVE'
```

### Process Multiple Batches

```bash
# Get list of batches ready to process
ready=$(sqlite3 /project/dash_agir/matthew.kutugata/pipeline_tracking.db \
    "SELECT batch_id FROM batches WHERE transfer_status='transferred' AND processing_status='pending'" | head -5)

# Process each one
for batch in $ready; do
    echo "Processing $batch..."
    ./scripts/workflow.sh process $batch
    sleep 10
done

# Monitor jobs
watch -n 30 'squeue -u $USER -A dash_agir'
```

## Common Workflows

### New Data from Field

```bash
# 1. Discover new batches
./scripts/workflow.sh check-missing

# 2. Sync them to JUNO
./scripts/workflow.sh sync-all

# 3. Wait for syncs (check globus task list)
# 4. Process each batch
./scripts/workflow.sh full-pipeline MD_2025-10-22
```

### Resume Failed Transfers

```bash
# Find failed batches
sqlite3 /project/dash_agir/matthew.kutugata/pipeline_tracking.db \
    "SELECT batch_id FROM batches WHERE ncsu_sync_status='failed'"

# Retry sync
./scripts/workflow.sh sync MD_2025-10-22

# Or for transfer failures:
sqlite3 /project/dash_agir/matthew.kutugata/pipeline_tracking.db \
    "SELECT batch_id FROM batches WHERE transfer_status='failed'"

./scripts/workflow.sh transfer MD_2025-10-22
```

### Verify Data Integrity

```bash
# Check if batch exists in all three locations
batch="MD_2025-10-22"

echo "NCSU:"
globus ls <NCSU_ENDPOINT>:<NCSU_PATH>/$batch | wc -l

echo "JUNO:"
globus ls 904c2108-90cf-11e8-9672-0a6d4e044368:/project/dash_agir/semifield-upload/$batch | wc -l

echo "Ceres:"
ls -1 /90daydata/dash_agir/data/semifield-upload/$batch/*.RAW | wc -l
```

## File Locations

| Component | Path |
|-----------|------|
| Workflow script | `~/repos/svs-raw-api/scripts/workflow.sh` |
| Globus manager | `~/repos/svs-raw-api/scripts/globus_manager.py` |
| DB manager | `~/repos/svs-raw-api/scripts/db_manager.py` |
| Database | `/project/dash_agir/matthew.kutugata/pipeline_tracking.db` |
| NCSU data | `<NCSU_PATH>/semifield-upload/` |
| JUNO data | `/project/dash_agir/semifield-upload/` |
| Ceres scratch | `/90daydata/dash_agir/data/semifield-upload/` |
| Final output | `/project/dash_agir/matthew.kutugata/semifield-developed-images/` |
| Logs | `/project/dash_agir/matthew.kutugata/logs/` |

## Status Values

### NCSU Sync Status
- `unknown` - Not yet checked
- `needed` - Exists in NCSU, needs sync to JUNO
- `syncing` - Globus transfer in progress
- `synced` - Successfully synced to JUNO
- `failed` - Sync failed

### Transfer Status (JUNO â†’ Ceres)
- `pending` - Not yet started
- `transferring` - Globus transfer in progress
- `transferred` - Successfully on Ceres
- `failed` - Transfer failed

### Processing Status
- `pending` - Not yet started
- `processing` - SLURM job running
- `completed` - Processing finished
- `failed` - Processing failed

## Troubleshooting

### Can't find NCSU batches

```bash
# Run configuration helper
./scripts/find_ncsu_endpoint.sh

# Verify endpoint is set
grep NCSU_ENDPOINT ~/repos/svs-raw-api/scripts/globus_manager.py
```

### Transfer stuck/failed

```bash
# Check Globus task
globus task show <task-id>

# Cancel if needed
globus task cancel <task-id>

# Retry
./scripts/workflow.sh sync <batch-id>
```

### Processing failed

```bash
# Check SLURM job output
sacct -j <job-id> --format=JobID,JobName,State,ExitCode

# View log
cat /project/dash_agir/matthew.kutugata/logs/snakemake_<batch-id>_<job-id>.out

# Retry
./scripts/workflow.sh process <batch-id>
```

### Database locked

```bash
# Check for open connections
lsof /project/dash_agir/matthew.kutugata/pipeline_tracking.db

# Force close
pkill -f pipeline_tracking.db

# Wait a moment, then retry
```

## Performance Expectations

| Stage | Time (100 images) | Parallelization |
|-------|-------------------|-----------------|
| NCSU â†’ JUNO sync | 5-15 min | 1 transfer |
| JUNO â†’ Ceres transfer | 5-10 min | 1 transfer |
| RAW â†’ DNG â†’ JPG | 20-30 min | 12 SLURM jobs |
| **Total** | **40-60 min** | |

## Support Resources

- **Globus docs**: https://docs.globus.org/
- **NCSU Research Storage**: https://research.oit.ncsu.edu/docs/storage/
- **SCINet Ceres**: https://scinet.usda.gov/guides/ceres/
- **This pipeline**: See THREE_TIER_SETUP.md

## Quick Checks

```bash
# Is Globus working?
globus whoami

# Can I access NCSU?
globus ls <NCSU_ENDPOINT>:<NCSU_PATH>

# Can I access JUNO?
globus ls 904c2108-90cf-11e8-9672-0a6d4e044368:/project/dash_agir

# Is database accessible?
sqlite3 /project/dash_agir/matthew.kutugata/pipeline_tracking.db "SELECT COUNT(*) FROM batches;"

# Is conda env working?
source /project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep/bin/activate
python --version
```
