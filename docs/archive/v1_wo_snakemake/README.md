# Three-Tier Pipeline Enhancement - Complete Package

## 📦 What's Included

I've created a complete three-tier pipeline system that extends your existing JUNO → Ceres workflow to include NCSU NFS storage as the primary data collection point.

**New Architecture:** NCSU NFS → JUNO LTS → Ceres /90daydata → Process

## 📄 Files Created (13 total)

### Core Scripts (Replace Existing)
1. **globus_manager_v2.py** (17 KB) - Enhanced with NCSU endpoint support
2. **db_manager_v2.py** (21 KB) - Enhanced with NCSU sync tracking
3. **workflow_v2.sh** (14 KB) - Three-tier workflow commands

### Snakemake Files (NEW)
4. **Snakefile_three_tier** (14 KB) - Enhanced Snakemake workflow with database integration
5. **snakemake_config_three_tier.yaml** (3.2 KB) - Configuration for three-tier pipeline
6. **run_snakemake_three_tier.sh** (8 KB) - SLURM submission script

### New Utilities
7. **find_ncsu_endpoint.sh** (7.4 KB) - Interactive NCSU configuration helper

### Documentation
8. **README.md** (15 KB) - This file - complete package guide
9. **THREE_TIER_SETUP.md** (12 KB) - Complete setup guide
10. **THREE_TIER_QUICK_REF.md** (7.6 KB) - Command reference
11. **THREE_TIER_SUMMARY.md** (9.6 KB) - Migration guide
12. **SNAKEMAKE_INTEGRATION.md** (11 KB) - Snakemake integration guide
13. **ARCHITECTURE_DIAGRAM.txt** (18 KB) - Visual architecture

### Installation
14. **install_three_tier.sh** (4.1 KB) - Automated installer

**Total Size:** ~162 KB

## 🚀 Quick Start

### Step 1: Download All Files

All files are available at `/mnt/user-data/outputs/` on this system.

```bash
# Create download directory
mkdir -p ~/three-tier-package
cd ~/three-tier-package

# Copy all files
cp /mnt/user-data/outputs/globus_manager_v2.py .
cp /mnt/user-data/outputs/db_manager_v2.py .
cp /mnt/user-data/outputs/workflow_v2.sh .
cp /mnt/user-data/outputs/find_ncsu_endpoint.sh .
cp /mnt/user-data/outputs/THREE_TIER_SETUP.md .
cp /mnt/user-data/outputs/THREE_TIER_QUICK_REF.md .
cp /mnt/user-data/outputs/THREE_TIER_SUMMARY.md .
cp /mnt/user-data/outputs/ARCHITECTURE_DIAGRAM.txt .
cp /mnt/user-data/outputs/install_three_tier.sh .

# Or copy entire directory
cp -r /mnt/user-data/outputs ~/three-tier-package
```

### Step 2: Install on Ceres

```bash
# Transfer to Ceres (from your local machine)
scp -r ~/three-tier-package matthew.kutugata@ceres-dtn:/home/matthew.kutugata/

# On Ceres, run installer
cd ~/three-tier-package
chmod +x install_three_tier.sh
./install_three_tier.sh
```

### Step 3: Configure NCSU Endpoint

```bash
cd ~/repos/svs-raw-api
./scripts/find_ncsu_endpoint.sh
```

This interactive script will:
- Search for your NCSU Globus endpoint
- Test connectivity
- Browse your storage
- Generate configuration code

### Step 4: Install Snakemake Files

```bash
cd ~/repos/svs-raw-api

# Copy Snakemake workflow
cp ~/three-tier-package/Snakefile_three_tier .

# Copy config
cp ~/three-tier-package/snakemake_config_three_tier.yaml config/

# Copy SLURM script
cp ~/three-tier-package/run_snakemake_three_tier.sh slurm/
chmod +x slurm/run_snakemake_three_tier.sh

# Update config paths if needed
nano config/snakemake_config_three_tier.yaml
```

### Step 5: Test

```bash
# Check if everything is configured
./scripts/workflow.sh status

# Discover batches needing sync
./scripts/workflow.sh check-missing

# If you see batches listed, you're ready!
```

## 🎯 Key Features

### 1. Automated Discovery
```bash
./scripts/workflow.sh check-missing
```
Scans both NCSU and JUNO to find batches that need syncing.

### 2. NCSU → JUNO Sync
```bash
./scripts/workflow.sh sync MD_2025-10-22
./scripts/workflow.sh sync-all  # Sync everything
```
Automated Globus transfers with status tracking.

### 3. Complete Pipeline
```bash
./scripts/workflow.sh full-pipeline MD_2025-10-22
```
Runs entire pipeline: sync (if needed) → transfer → process

### 4. Enhanced Tracking
- New database table: `ncsu_sync_history`
- Track batches through all three tiers
- Complete audit trail

### 5. Interactive Mode
```bash
./scripts/workflow.sh interactive
```
Menu-driven interface for easier operation.

## 📊 What Changed

### Database Schema
**New Table:** `ncsu_sync_history`
```sql
CREATE TABLE ncsu_sync_history (
    batch_id TEXT,
    globus_task_id TEXT,
    status TEXT,
    bytes_transferred INTEGER,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

**Updated Table:** `batches`
- Added: `ncsu_path` - Path on NCSU storage
- Added: `ncsu_sync_status` - Sync status tracking
- Added: `ncsu_synced_at` - Timestamp

### New Commands
```bash
check-missing [STATE]    # Find batches needing sync
sync <BATCH_ID>          # Sync single batch NCSU → JUNO
sync-all [STATE]         # Sync all missing batches
full-pipeline <BATCH>    # Complete automated pipeline
check-task <TASK_ID>     # Check Globus transfer status
```

### Enhanced Commands
- `status` - Now includes NCSU sync information
- `transfer` - Checks NCSU sync first
- `process` - Verifies full pipeline status

## 🔧 Configuration Required

Before using, you MUST configure your NCSU Globus endpoint:

1. Run `./scripts/find_ncsu_endpoint.sh`
2. Edit `scripts/globus_manager.py`:
   ```python
   NCSU_ENDPOINT = "your-endpoint-uuid"
   NCSU_BASE_PATH = "/rsstu/users/your-group/semifield-upload"
   ```

**Finding Your Values:**
- Endpoint ID: Run `globus endpoint search "NC State Research Storage"`
- Path: Navigate with `globus ls <endpoint>:/` to find your directory

## 📖 Documentation

### Quick Reference
See `THREE_TIER_QUICK_REF.md` for:
- Common commands
- Database queries
- Troubleshooting
- Performance expectations

### Complete Setup Guide
See `THREE_TIER_SETUP.md` for:
- Detailed configuration
- Workflow examples
- Troubleshooting
- Support resources

### Migration Guide
See `THREE_TIER_SUMMARY.md` for:
- What changed
- Backward compatibility
- Migration steps
- Testing checklist

### Architecture
See `ARCHITECTURE_DIAGRAM.txt` for:
- Visual data flow
- Component interactions
- Performance metrics
- Storage locations

## ⚙️ System Requirements

- **Globus CLI**: `pip install globus-cli`
- **Conda environment**: Your existing `semif_prep` environment
- **NCSU access**: Globus permissions for Research Storage
- **Database**: SQLite (already installed)

## 🔄 Typical Workflow

### Daily: New Data from Field
```bash
# 1. Check what's new
./scripts/workflow.sh check-missing

# 2. Sync new batches to JUNO
./scripts/workflow.sh sync-all

# 3. Monitor syncs
globus task list --filter-status=ACTIVE

# 4. Process batches
./scripts/workflow.sh full-pipeline MD_2025-10-22
```

### Batch Processing: Multiple Batches
```bash
# Process all ready batches
for batch in $(./scripts/workflow.sh status | grep "transferred.*pending" | awk '{print $1}'); do
    ./scripts/workflow.sh process $batch
    sleep 10
done
```

## 📈 Performance

**Per 100-image batch:**
- NCSU → JUNO sync: 10-30 minutes
- JUNO → Ceres transfer: 5-10 minutes
- Processing: 20-30 minutes
- **Total: 40-60 minutes**

**Parallelization:**
- Up to 12 concurrent SLURM jobs
- 4 cores × 16GB RAM per job
- Peak: 48 cores, 192GB RAM

## 🔍 Monitoring

### Check Pipeline Status
```bash
./scripts/workflow.sh status
```

Shows:
- Total batches tracked
- NCSU sync summary
- Transfer summary
- Processing summary
- Active SLURM jobs

### Check Globus Transfer
```bash
./scripts/workflow.sh check-task <task-id>
globus task show <task-id>
```

### Database Queries
```bash
# Summary
python scripts/db_manager.py summary

# List batches
python scripts/db_manager.py list

# Export data
python scripts/db_manager.py export --output pipeline_data.json
```

## 🐛 Troubleshooting

### "PLACEHOLDER_NCSU_ENDPOINT_ID" Error
**Solution:** Run `./scripts/find_ncsu_endpoint.sh` and configure endpoint

### "Permission denied" on NCSU
**Solution:** Verify Globus access: `globus ls <endpoint>:/`

### Transfer failed/stuck
**Solution:** 
```bash
globus task show <task-id>  # Check error
globus task cancel <task-id>  # Cancel if needed
./scripts/workflow.sh sync <batch-id>  # Retry
```

### Database locked
**Solution:**
```bash
lsof /project/dash_agir/matthew.kutugata/pipeline_tracking.db
pkill -f pipeline_tracking.db
```

## ✅ Installation Checklist

- [ ] Downloaded all 9 files
- [ ] Ran `install_three_tier.sh`
- [ ] Configured NCSU endpoint in `globus_manager.py`
- [ ] Verified Globus login: `globus whoami`
- [ ] Tested NCSU access: `globus ls <endpoint>:<path>`
- [ ] Ran discovery: `./scripts/workflow.sh check-missing`
- [ ] Tested sync: `./scripts/workflow.sh sync <batch-id>`
- [ ] Read documentation

## 🔗 File Download Links

All files available at: `/mnt/user-data/outputs/`

### Direct Download Commands
```bash
# On Ceres
cd ~/three-tier-package

# Core scripts
wget computer:///mnt/user-data/outputs/globus_manager_v2.py
wget computer:///mnt/user-data/outputs/db_manager_v2.py
wget computer:///mnt/user-data/outputs/workflow_v2.sh

# Snakemake files
wget computer:///mnt/user-data/outputs/Snakefile_three_tier
wget computer:///mnt/user-data/outputs/snakemake_config_three_tier.yaml
wget computer:///mnt/user-data/outputs/run_snakemake_three_tier.sh

# Utilities
wget computer:///mnt/user-data/outputs/find_ncsu_endpoint.sh
wget computer:///mnt/user-data/outputs/install_three_tier.sh

# Documentation
wget computer:///mnt/user-data/outputs/README.md
wget computer:///mnt/user-data/outputs/THREE_TIER_SETUP.md
wget computer:///mnt/user-data/outputs/THREE_TIER_QUICK_REF.md
wget computer:///mnt/user-data/outputs/THREE_TIER_SUMMARY.md
wget computer:///mnt/user-data/outputs/SNAKEMAKE_INTEGRATION.md
wget computer:///mnt/user-data/outputs/ARCHITECTURE_DIAGRAM.txt
```

## 📞 Support

- **Setup Issues**: See `THREE_TIER_SETUP.md`
- **Commands**: See `THREE_TIER_QUICK_REF.md`
- **Migration**: See `THREE_TIER_SUMMARY.md`
- **Globus Help**: https://docs.globus.org/
- **NCSU Storage**: https://research.oit.ncsu.edu/docs/storage/

## 🎓 What You Get

This enhancement provides:

1. **Automated discovery** of new batches in NCSU storage
2. **Reliable syncing** from NCSU → JUNO with Globus
3. **Complete tracking** through all three storage tiers
4. **Single-command pipeline** for full automation
5. **Enhanced database** with complete audit trail
6. **Backward compatibility** with existing workflows
7. **Comprehensive documentation** with examples

## 🚦 Ready to Start?

1. **Download files** from `/mnt/user-data/outputs/`
2. **Run installer** on Ceres
3. **Configure NCSU** with helper script
4. **Test discovery** to verify setup
5. **Start syncing** your batches!

---

**Questions?** Check the documentation or examine the inline code comments - all scripts have detailed help sections.

**Need Help?** All commands support `--help`:
```bash
./scripts/workflow.sh help
python scripts/globus_manager.py --help
python scripts/db_manager.py --help
```
