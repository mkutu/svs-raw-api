# Three-Tier Pipeline Enhancement Summary

## What Changed

Your pipeline has been enhanced to support a three-tier storage system:

**OLD:** JUNO → Ceres → Process

**NEW:** NCSU → JUNO → Ceres → Process

## Why This Matters

1. **Primary Data Location**: Field equipment uploads directly to NCSU NFS storage
2. **Archive Storage**: JUNO serves as long-term reliable archive
3. **Processing Efficiency**: Ceres /90daydata provides fast scratch space for compute
4. **Automated Sync**: Pipeline tracks and syncs data through all three tiers

## New Capabilities

### 1. Discovery
- Scan NCSU storage for new batches
- Compare NCSU vs JUNO to find missing data
- Track which batches need syncing

### 2. NCSU → JUNO Sync
- Globus transfers from NCSU to JUNO archive
- Automatic status tracking
- Resume capability for failed transfers

### 3. Enhanced Database
- New table: `ncsu_sync_history`
- New field: `ncsu_sync_status` in batches table
- New field: `ncsu_path` to track original location
- Complete audit trail through all tiers

### 4. Unified Workflow
- Single command to run entire pipeline: `./scripts/workflow.sh full-pipeline <batch-id>`
- Interactive menu for easier operation
- Batch status tracking across all stages

## Files Created/Updated

### Core Scripts

1. **globus_manager.py** (UPDATED)
   - Added NCSU endpoint support
   - New functions: `list_ncsu_batches()`, `find_missing_batches()`, `sync_ncsu_to_juno()`
   - Supports three-tier transfers

2. **db_manager.py** (UPDATED)
   - New `ncsu_sync_history` table
   - Enhanced `batches` table with NCSU tracking
   - New queries for NCSU sync status

3. **workflow.sh** (UPDATED)
   - New commands: `check-missing`, `sync`, `sync-all`, `full-pipeline`
   - Interactive menu mode
   - Color-coded output for clarity

### New Utilities

4. **find_ncsu_endpoint.sh** (NEW)
   - Interactive helper to find your NCSU Globus endpoint
   - Tests connectivity and permissions
   - Generates configuration code

### Documentation

5. **THREE_TIER_SETUP.md** (NEW)
   - Complete setup guide
   - Configuration instructions
   - Troubleshooting

6. **THREE_TIER_QUICK_REF.md** (NEW)
   - Quick reference for common commands
   - Database queries
   - Performance expectations

## Installation Steps

### 1. Update Your Files

Replace the following files in your existing repo:

```bash
cd ~/repos/svs-raw-api

# Backup originals
cp scripts/globus_manager.py scripts/globus_manager.py.backup
cp scripts/db_manager.py scripts/db_manager.py.backup
cp scripts/workflow.sh scripts/workflow.sh.backup

# Copy new versions (from /mnt/user-data/outputs/)
cp /path/to/globus_manager_v2.py scripts/globus_manager.py
cp /path/to/db_manager_v2.py scripts/db_manager.py
cp /path/to/workflow_v2.sh scripts/workflow.sh

# Add new files
cp /path/to/find_ncsu_endpoint.sh scripts/
cp /path/to/THREE_TIER_SETUP.md docs/
cp /path/to/THREE_TIER_QUICK_REF.md docs/

# Make executable
chmod +x scripts/*.sh
```

### 2. Configure NCSU Endpoint

```bash
# Run configuration helper
./scripts/find_ncsu_endpoint.sh

# This will:
# 1. Search for NC State endpoints
# 2. Test connectivity
# 3. Browse directories
# 4. Generate configuration code

# Then edit scripts/globus_manager.py with the provided values:
# NCSU_ENDPOINT = "your-endpoint-uuid"
# NCSU_BASE_PATH = "/rsstu/users/your-group/semifield-upload"
```

### 3. Update Database Schema

The database will automatically update when you run commands, but you can also manually:

```bash
# Initialize with new schema
python scripts/db_manager.py --db /project/dash_agir/matthew.kutugata/pipeline_tracking.db init

# Verify new tables exist
sqlite3 /project/dash_agir/matthew.kutugata/pipeline_tracking.db ".tables"
# Should see: batches, files, ncsu_sync_history, transfer_history, processing_history
```

### 4. Test the Setup

```bash
# Check prerequisites
./scripts/workflow.sh status

# Test NCSU discovery
./scripts/workflow.sh check-missing

# If you see batches listed, configuration is correct!
```

## Database Schema Changes

### New Table: ncsu_sync_history

```sql
CREATE TABLE ncsu_sync_history (
    id INTEGER PRIMARY KEY,
    batch_id TEXT NOT NULL,
    globus_task_id TEXT,
    status TEXT NOT NULL,
    bytes_transferred INTEGER,
    files_transferred INTEGER,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
);
```

### Updated Table: batches

New fields added:
- `ncsu_path TEXT` - Path on NCSU storage
- `ncsu_sync_status TEXT` - Status of NCSU → JUNO sync
- `ncsu_synced_at TIMESTAMP` - When sync completed

## Workflow Changes

### Before (Two-Tier)

```bash
# Old workflow
1. Manually ensure data is in JUNO
2. ./scripts/workflow.sh discover
3. ./scripts/workflow.sh process <batch-id>
```

### After (Three-Tier)

```bash
# New workflow - automated discovery and sync
1. ./scripts/workflow.sh check-missing
2. ./scripts/workflow.sh sync-all  # (if needed)
3. ./scripts/workflow.sh full-pipeline <batch-id>

# Or just:
./scripts/workflow.sh full-pipeline <batch-id>
# (automatically handles NCSU sync if needed)
```

## Command Reference

### New Commands

```bash
# Discovery
./scripts/workflow.sh check-missing [STATE]  # Find batches in NCSU but not JUNO

# Syncing
./scripts/workflow.sh sync <BATCH_ID>        # Sync single batch NCSU → JUNO
./scripts/workflow.sh sync-all [STATE]       # Sync all missing batches

# Complete pipeline
./scripts/workflow.sh full-pipeline <BATCH_ID>  # NCSU → JUNO → Ceres → Process

# Monitoring
./scripts/workflow.sh check-task <TASK_ID>   # Check Globus transfer status
```

### Updated Commands

```bash
# Status now includes NCSU sync info
./scripts/workflow.sh status

# Transfer now checks if NCSU sync needed first
./scripts/workflow.sh transfer <BATCH_ID>

# Process now checks full pipeline status
./scripts/workflow.sh process <BATCH_ID>
```

## Configuration Requirements

Before using the three-tier system, you MUST configure:

### 1. NCSU Globus Endpoint

Edit `scripts/globus_manager.py`:

```python
class GlobusTransferManager:
    # UPDATE THESE:
    NCSU_ENDPOINT = "your-endpoint-uuid-here"
    NCSU_BASE_PATH = "/rsstu/users/your-group/semifield-upload"
    
    # These are already set:
    JUNO_ENDPOINT = "904c2108-90cf-11e8-9672-0a6d4e044368"
    CERES_ENDPOINT = "f45a24f8-09ba-11ec-b342-1feaf93e3729"
```

Use `./scripts/find_ncsu_endpoint.sh` to find your values.

### 2. Verify Globus Access

```bash
# Login to Globus
globus login

# Test NCSU access
globus ls <NCSU_ENDPOINT>:<NCSU_PATH>

# Should see list of batch directories
```

## Migration from Two-Tier System

If you have existing batches tracked in the old database:

### Option 1: Keep Existing Data

The new schema is backward compatible. Existing batches will:
- Have `ncsu_path` set to NULL (skips NCSU sync)
- Continue processing normally from JUNO

### Option 2: Add NCSU Paths to Existing Batches

```sql
-- Update existing batches with NCSU paths
UPDATE batches 
SET ncsu_path = '<NCSU_PATH>/' || batch_id,
    ncsu_sync_status = 'unknown'
WHERE ncsu_path IS NULL;

-- Then check which need syncing
SELECT batch_id FROM batches WHERE ncsu_sync_status = 'unknown';
```

## Performance Impact

**No negative impact on existing workflows:**
- JUNO → Ceres transfers work exactly as before
- Processing time unchanged
- Database operations remain fast

**New capabilities add:**
- NCSU → JUNO sync: ~5-15 min per batch (one-time per batch)
- Automated discovery: < 1 minute
- Enhanced tracking: negligible overhead

## Backward Compatibility

✅ **Fully backward compatible**

- Old commands still work: `discover`, `transfer`, `process`
- Existing batches (without NCSU paths) process normally
- New features are opt-in via new commands

## Testing Checklist

- [ ] Globus CLI installed and logged in
- [ ] NCSU endpoint configured in globus_manager.py
- [ ] Can list NCSU batches: `./scripts/workflow.sh check-missing`
- [ ] Database updated with new schema
- [ ] Test sync: `./scripts/workflow.sh sync <test-batch-id>`
- [ ] Monitor transfer: `globus task show <task-id>`
- [ ] Test full pipeline: `./scripts/workflow.sh full-pipeline <batch-id>`

## Troubleshooting

### "PLACEHOLDER_NCSU_ENDPOINT_ID" Error

**Problem:** NCSU endpoint not configured

**Solution:** 
```bash
./scripts/find_ncsu_endpoint.sh  # Find your endpoint
# Edit scripts/globus_manager.py with values
```

### "Permission denied" on NCSU

**Problem:** Don't have Globus access to NCSU storage

**Solution:**
- Verify with: `globus ls <NCSU_ENDPOINT>:/`
- Contact NCSU OIT to request access
- Check if you need to be in a specific group

### "Batch not found" in NCSU

**Problem:** Batch exists in JUNO but not NCSU

**Solution:** This is normal! Batches don't need to be in NCSU if already in JUNO.
Set `ncsu_path` to NULL in database for these batches:

```sql
UPDATE batches SET ncsu_path = NULL WHERE batch_id = 'MD_2025-10-22';
```

## Support and Documentation

- **Setup Guide**: `THREE_TIER_SETUP.md`
- **Quick Reference**: `THREE_TIER_QUICK_REF.md`
- **Code Comments**: All scripts have detailed inline comments
- **Database Schema**: See `db_manager.py` for table definitions

## Next Steps

1. **Install files** - Copy new versions to your repo
2. **Configure NCSU** - Run `find_ncsu_endpoint.sh`
3. **Test discovery** - Run `./scripts/workflow.sh check-missing`
4. **Sync test batch** - Try `./scripts/workflow.sh sync <batch-id>`
5. **Monitor** - Check `globus task show <task-id>`
6. **Full pipeline** - Run `./scripts/workflow.sh full-pipeline <batch-id>`

## Questions?

See documentation or check the inline code comments. All scripts have detailed help:

```bash
./scripts/workflow.sh help
python scripts/globus_manager.py --help
python scripts/db_manager.py --help
```
