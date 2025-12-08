#!/usr/bin/env bash
#SBATCH --job-name=svs_process
#SBATCH --account=dash_agir
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=/project/dash_agir/logs/svs_process_%j.out
#SBATCH --error=/project/dash_agir/logs/svs_process_%j.err

set -euo pipefail

echo "[$(date)] Starting SVS RAW processing pipeline"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Pipeline stage: ${PIPELINE_STAGE:-raw_to_dng}"

# ============================================================
# Environment Setup
# ============================================================

# Load PostgreSQL module
module load postgresql
echo "[INIT] Loading database connection parameters..."
source /project/dash_agir/postgres/pg_coords.env

# Load Python environment
module load miniconda
source activate /project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep

# Install required packages if not present
pip install -e /project/dash_agir/matthew.kutugata/repos/svs-raw-api >/dev/null 2>&1 || true

# ============================================================
# Configuration
# ============================================================

REPO_DIR="/project/dash_agir/matthew.kutugata/repos/svs-raw-api"
CONFIG="${REPO_DIR}/config/scinet.yaml"
PYTHON_SCRIPT="${REPO_DIR}/scripts/process_batch_integrated.py"

# Pipeline stage: raw_to_dng or dng_to_jpg
PIPELINE_STAGE="${PIPELINE_STAGE:-raw_to_dng}"

# Number of batches to process in this job
BATCH_LIMIT="${BATCH_LIMIT:-10}"

# Optional: specific batch ID to process
BATCH_ID="${BATCH_ID:-}"

# ============================================================
# Verify Database Connection
# ============================================================

echo "[INFO] Testing database connection..."
psql -c "SELECT version();" >/dev/null 2>&1 || {
    echo "[ERROR] Cannot connect to database"
    echo "[ERROR] PGHOST=${PGHOST}"
    echo "[ERROR] PGPORT=${PGPORT}"
    echo "[ERROR] PGDATABASE=${PGDATABASE}"
    exit 1
}
echo "[INFO] Database connection OK"

# ============================================================
# Run Processing Pipeline
# ============================================================

echo "[INFO] Starting ${PIPELINE_STAGE} pipeline..."
echo "[INFO] Configuration: ${CONFIG}"
echo "[INFO] Batch limit: ${BATCH_LIMIT}"

if [[ -n "${BATCH_ID}" ]]; then
    echo "[INFO] Processing specific batch: ${BATCH_ID}"
    python3 "${PYTHON_SCRIPT}" \
        --config "${CONFIG}" \
        --stage "${PIPELINE_STAGE}" \
        --batch-id "${BATCH_ID}" \
        --job-id "${SLURM_JOB_ID}" \
        --log-level INFO
else
    echo "[INFO] Processing up to ${BATCH_LIMIT} batches from queue"
    python3 "${PYTHON_SCRIPT}" \
        --config "${CONFIG}" \
        --stage "${PIPELINE_STAGE}" \
        --limit "${BATCH_LIMIT}" \
        --job-id "${SLURM_JOB_ID}" \
        --log-level INFO
fi

EXIT_CODE=$?

# ============================================================
# Report Status
# ============================================================

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "[SUCCESS] Pipeline completed successfully"
else
    echo "[ERROR] Pipeline failed with exit code ${EXIT_CODE}"
fi

echo "[$(date)] Job complete"
exit ${EXIT_CODE}
