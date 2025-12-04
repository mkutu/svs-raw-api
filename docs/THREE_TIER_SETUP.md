# Three-Tier Pipeline Setup Guide
## NCSU → JUNO → Ceres → Process

## Overview

This enhanced pipeline manages data flow through three storage tiers:

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  NCSU NFS    │  →   │  JUNO LTS    │  →   │ Ceres Scratch│  →   │   Process    │
│  (Upload)    │      │  (Archive)   │      │ (/90daydata) │      │ (Convert)    │
└──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
     Primary             Long-term           Processing            Final outputs
     storage             storage             temporary             to JUNO/Project
```

### Why Three Tiers?

1. **NCSU NFS**: Primary data collection location, easily accessible to field equipment
2. **JUNO LTS**: Long-term archive storage, reliable and backed up
3. **Ceres Scratch**: Fast processing space with compute resources

## Prerequisites

### 1. Globus CLI Setup

```bash
# On Ceres
pip install globus-cli --break-system-packages

# Login to Globus
globus login

# Verify login
globus whoami
```

### 2. Find Your NCSU Globus Endpoint

You need to find the specific Globus endpoint and path for your NCSU storage:

```bash
# Search for NC State endpoints
globus endpoint search "NC State"

# Example output:
# ID: <endpoint-uuid>
# Display Name: NC State Research Storage
```

**You'll need:**
- Endpoint ID (UUID)
- Path to your data (e.g., `/rsstu/users/[group]/semifield-upload`)

### 3. Test NCSU Access

```bash
# Test listing directory (replace with your endpoint and path)
globus ls <NCSU_ENDPOINT_ID>:/rsstu/users/[your-group]/

# Expected output: list of directories/files
```

## Configuration

### Step 1: Update Globus Manager with NCSU Details

Edit `scripts/globus_manager.py`:

```python
class GlobusTransferManager:
    # ... existing code ...
    
    # UPDATE THESE TWO LINES:
    NCSU_ENDPOINT = "YOUR_NCSU_ENDPOINT_ID_HERE"
    NCSU_BASE_PATH = "/rsstu/users/YOUR_GROUP/semifield-upload"
    
    # Leave these as-is:
    JUNO_ENDPOINT = "904c2108-90cf-11e8-9672-0a6d4e044368"
    CERES_ENDPOINT = "f45a24f8-09ba-11ec-b342-1feaf93e3729"
```

**To find your values:**

```bash
# 1. Find NCSU endpoint
globus endpoint search "NC State Research Storage"
# Copy the UUID

# 2. Find your path
globus ls <NCSU_ENDPOINT>:/
# Navigate to find your semifield-upload directory
```

### Step 2: Verify Configuration

```bash
# Test NCSU listing
cd ~/repos/svs-raw-api
source /project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep/bin/activate

python scripts/globus_manager.py check-missing
```

**Expected output:**
```
🔍 Scanning NCSU storage: /rsstu/users/[group]/semifield-upload
🔍 Scanning JUNO archive: /project/dash_agir/semifield-upload

📊 Summary:
   NCSU batches:  [number]
   JUNO batches:  [number]
   Missing in JUNO: [number]
```

If you see an error about "PLACEHOLDER_NCSU_ENDPOINT_ID", you need to complete Step 1.

## Database Schema Updates

The database now tracks three storage locations:

```sql
-- Batches table with NCSU tracking
CREATE TABLE batches (
    batch_id TEXT PRIMARY KEY,
    
    -- Storage paths (three tiers)
    ncsu_path TEXT,                    -- Path on NCSU NFS
    juno_path TEXT NOT NULL,           -- Path on JUNO
    ceres_path TEXT,                   -- Path on Ceres
    
    -- Status tracking
    ncsu_sync_status TEXT,             -- unknown, needed, syncing, synced, failed
    transfer_status TEXT,              -- pending, transferring, transferred, failed
    processing_status TEXT,            -- pending, processing, completed, failed
    
    -- ... other fields
);

-- New table: NCSU sync history
CREATE TABLE ncsu_sync_history (
    id INTEGER PRIMARY KEY,
    batch_id TEXT NOT NULL,
    globus_task_id TEXT,
    status TEXT NOT NULL,
    bytes_transferred INTEGER,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
);
```

## Workflow

### Discovery Phase: Find Missing Batches

```bash
# Check what needs to be synced from NCSU to JUNO
./scripts/workflow.sh check-missing

# Filter by state
./scripts/workflow.sh check-missing MD
```

**What it does:**
1. Lists all batches in NCSU storage
2. Lists all batches in JUNO storage
3. Shows batches present in NCSU but missing from JUNO

### Sync Phase: NCSU → JUNO

```bash
# Sync single batch
./scripts/workflow.sh sync MD_2025-10-22

# Sync all missing batches
./scripts/workflow.sh sync-all

# Sync all for specific state
./scripts/workflow.sh sync-all MD
```

**What it does:**
1. Submits Globus transfer task from NCSU → JUNO
2. Updates database with sync status
3. Records task ID for monitoring

**Monitor sync:**
```bash
# Get task ID from output, then:
globus task show <task-id>

# Or use workflow command:
./scripts/workflow.sh check-task <task-id>
```

### Transfer Phase: JUNO → Ceres

```bash
# Transfer single batch (must be in JUNO first)
./scripts/workflow.sh transfer MD_2025-10-22
```

**What it does:**
1. Submits Globus transfer from JUNO → Ceres /90daydata
2. Updates database with transfer status

### Processing Phase: Convert Images

```bash
# Process batch (must be on Ceres first)
./scripts/workflow.sh process MD_2025-10-22
```

**What it does:**
1. Submits SLURM job for RAW → DNG → JPG conversion
2. Uses Snakemake for parallel processing
3. Updates database with processing status

### Full Pipeline: All Stages

```bash
# Run complete pipeline for one batch
./scripts/workflow.sh full-pipeline MD_2025-10-22
```

**What it does:**
1. Checks if batch needs NCSU sync (does it if needed)
2. Transfers JUNO → Ceres
3. Processes on Ceres
4. Updates database throughout

## Interactive Mode

For easier operation:

```bash
./scripts/workflow.sh interactive
```

This provides a menu-driven interface for all operations.

## Monitoring

### Check Overall Status

```bash
./scripts/workflow.sh status
```

**Shows:**
- Total batches tracked
- NCSU sync status summary
- Transfer status summary
- Processing status summary
- Recent SLURM jobs

### Database Queries

```bash
# Summary
python scripts/db_manager.py --db /project/dash_agir/matthew.kutugata/pipeline_tracking.db summary

# List all batches
python scripts/db_manager.py --db /project/dash_agir/matthew.kutugata/pipeline_tracking.db list

# Export to JSON
python scripts/db_manager.py --db /project/dash_agir/matthew.kutugata/pipeline_tracking.db export --output pipeline_data.json
```

### View Specific Tables

```bash
sqlite3 /project/dash_agir/matthew.kutugata/pipeline_tracking.db

# In SQLite shell:
-- See batches needing sync
SELECT batch_id, ncsu_sync_status, transfer_status 
FROM batches 
WHERE ncsu_sync_status IN ('needed', 'unknown')
ORDER BY batch_id;

-- See recent NCSU syncs
SELECT * FROM ncsu_sync_history 
ORDER BY started_at DESC 
LIMIT 10;

-- See complete pipeline status
SELECT 
    batch_id,
    ncsu_sync_status,
    transfer_status,
    processing_status,
    ncsu_synced_at,
    transferred_at,
    processing_completed_at
FROM batches
ORDER BY date DESC;
```

## Typical Workflows

### Daily Workflow: New Data from NCSU

```bash
# 1. Check what's new
./scripts/workflow.sh check-missing

# 2. Sync all new batches to JUNO
./scripts/workflow.sh sync-all

# 3. Wait for syncs to complete (check tasks)
globus task list --filter-status=ACTIVE

# 4. Once synced, process each batch
./scripts/workflow.sh transfer MD_2025-10-22
./scripts/workflow.sh process MD_2025-10-22

# OR: Use full-pipeline for each batch
./scripts/workflow.sh full-pipeline MD_2025-10-22
```

### Batch Processing: Multiple Batches

```bash
# Get list of batches needing sync
batches=$(./scripts/workflow.sh check-missing | grep "• " | awk '{print $2}')

# Sync them all
for batch in $batches; do
    ./scripts/workflow.sh sync $batch
    sleep 5  # Stagger submissions
done

# Monitor all transfers
globus task list --filter-status=ACTIVE

# Once complete, process
for batch in $batches; do
    ./scripts/workflow.sh full-pipeline $batch
done
```

## Troubleshooting

### "Not logged into Globus"

```bash
globus login
# Follow prompts to authenticate
```

### "NCSU endpoint not configured"

Edit `scripts/globus_manager.py` and replace:
```python
NCSU_ENDPOINT = "PLACEHOLDER_NCSU_ENDPOINT_ID"
NCSU_BASE_PATH = "PLACEHOLDER_NCSU_PATH"
```

With your actual values.

### "Permission denied" on NCSU

Verify you have access:
```bash
globus ls <NCSU_ENDPOINT>:/rsstu/users/[your-group]/
```

If this fails, contact NCSU OIT to verify your Globus permissions.

### Transfer Stuck

Check task status:
```bash
globus task show <task-id>

# If failed, see details:
globus task show <task-id> --format json | jq '.nice_status_details'

# Cancel if needed:
globus task cancel <task-id>
```

### Database Issues

```bash
# Reinitialize database (WARNING: loses data)
rm /project/dash_agir/matthew.kutugata/pipeline_tracking.db
python scripts/db_manager.py --db /project/dash_agir/matthew.kutugata/pipeline_tracking.db init

# Verify database
sqlite3 /project/dash_agir/matthew.kutugata/pipeline_tracking.db ".tables"
```

## File Organization

```
~/repos/svs-raw-api/
├── scripts/
│   ├── globus_manager.py      # Enhanced with NCSU support
│   ├── db_manager.py           # Enhanced with NCSU tracking
│   └── workflow.sh             # Three-tier workflow manager
├── config/
│   └── snakemake_config_enhanced.yaml
├── slurm/
│   └── run_snakemake_enhanced.sh
└── Snakefile_enhanced
```

## Storage Locations

| Tier | Location | Purpose | Capacity | Speed |
|------|----------|---------|----------|-------|
| NCSU | `/rsstu/users/[group]/semifield-upload` | Primary upload | Large | Medium |
| JUNO | `/project/dash_agir/semifield-upload` | Long-term archive | Very Large | Slow |
| Ceres | `/90daydata/dash_agir/data/semifield-upload` | Processing scratch | Limited (90 days) | Fast |

## Data Retention

- **NCSU**: Permanent (or per lab policy)
- **JUNO**: Permanent archive
- **Ceres /90daydata**: 90 days, then auto-deleted
- **Final outputs**: Moved to `/project/dash_agir/matthew.kutugata/semifield-developed-images`

## Performance

**NCSU → JUNO Transfer:**
- Speed: ~100-500 MB/s (varies by time of day)
- 100-image batch (~20GB): 5-15 minutes

**JUNO → Ceres Transfer:**
- Speed: ~200-800 MB/s
- 100-image batch: 5-10 minutes

**Processing:**
- RAW → DNG → JPG: 20-30 minutes per 100 images
- Parallelized across 12 SLURM jobs

**Total pipeline time per batch:** 40-60 minutes

## Next Steps

1. **Configure NCSU endpoint** (see Configuration section)
2. **Test discovery:** `./scripts/workflow.sh check-missing`
3. **Test sync:** `./scripts/workflow.sh sync <batch-id>`
4. **Monitor:** `globus task list`
5. **Process:** `./scripts/workflow.sh full-pipeline <batch-id>`

## Support

- Globus issues: https://docs.globus.org/
- NCSU storage: OIT Research Services
- SCINet Ceres: scinet_vrsc@usda.gov
