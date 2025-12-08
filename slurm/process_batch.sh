#!/usr/bin/env bash

# ============================================================
#                   IMPORTS & SETUP
# ============================================================

module load postgresql
echo "[INIT] Loading database connection parameters..."
source /project/dash_agir/postgres/pg_coords.env
module load miniconda
source activate /project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep

PSQL="psql -v ON_ERROR_STOP=1 -h $PGHOST -p $PGPORT -d $PGDATABASE -U $PGUSER"


# ============================================================
#               PATH DEFINITIONS
# ============================================================
PYTHON_SCRIPT="/project/dash_agir/matthew.kutugata/repos/svs-raw-api/slurm/process_batch.py"

CONFIG="config/scinet.yaml"
