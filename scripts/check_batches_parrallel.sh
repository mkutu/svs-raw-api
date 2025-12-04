#!/usr/bin/env bash
#SBATCH --job-name=semifield_globus_scan
#SBATCH --account=dash_agir
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16          # match N_JOBS in the script
#SBATCH --mem=4G                    # this job is I/O bound, 4G is plenty
#SBATCH --time=02:00:00             # adjust based on how big the dirs are
#SBATCH --partition=general         # <-- change to the right partition on SciNet
#SBATCH --output=/project/dash_agir/logs/check_batches_parrallel_%x_%j.out.log
#SBATCH --error=/project/dash_agir/logs/check_batches_parrallel_%x_%j.err.log

set -euo pipefail

echo "[INFO] Starting job on $(hostname) at $(date)"
echo "[INFO] SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

# -----------------------------
# Load env with globus-cli
# -----------------------------
# Example if you use conda/miniforge:
module load miniconda || true

# adjust this path/env name to whatever you actually use
source activate /project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep 2>/dev/null || \
conda activate semif_prep 2>/dev/null || \
echo "[WARN] Could not activate conda env; make sure globus-cli is on PATH"

# Optional: override N_JOBS from Slurm CPUs
export N_JOBS="${SLURM_CPUS_PER_TASK:-16}"



# -------------------------
# TUNABLE PARALLELISM
# -------------------------
# How many batches to process in parallel per location
N_JOBS=16

# -------------------------
# Endpoints
# -------------------------
JUNO_EP="904c2108-90cf-11e8-9672-0a6d4e044368"
NCSU_1_EP="2f7f6170-8d5c-11e9-8e6a-029d279f7e24"
NCSU_2_EP="2f7f6170-8d5c-11e9-8e6a-029d279f7e24"
NCSU_3_EP="2f7f6170-8d5c-11e9-8e6a-029d279f7e24"

# -------------------------
# Paths
# -------------------------
NCSU_ROOT="/rsstu/users/s/screberg"

# RAWS (uploads)
NCSU_RAW_PATH_1="$NCSU_ROOT/longterm_images2/semifield-upload"
NCSU_RAW_PATH_2="$NCSU_ROOT/longterm_images/semifield-upload"
NCSU_RAW_PATH_3="$NCSU_ROOT/GROW_DATA/semifield-upload"
JUNO_RAW_PATH="/LTS/project/dash_agir/semifield-upload"

# JPGS / developed
NCSU_DEV_PATH_1="$NCSU_ROOT/longterm_images2/semifield-developed-images"
NCSU_DEV_PATH_2="$NCSU_ROOT/longterm_images/semifield-developed-images"
NCSU_DEV_PATH_3="$NCSU_ROOT/GROW_DATA/semifield-developed-images"
JUNO_DEV_PATH="/LTS/project/dash_agir/semifield-developed-images"

LOGS_DIR="/project/dash_agir/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_CSV="$LOGS_DIR/batch_summary_${TIMESTAMP}.csv"

mkdir -p "$LOGS_DIR"

# -------------------------
# Arrays for locations
# -------------------------

# Upload (RAW) locations
RAW_ENDPOINTS=(
  "$JUNO_EP"
  "$NCSU_1_EP"
  "$NCSU_2_EP"
  "$NCSU_3_EP"
)

RAW_PATHS=(
  "$JUNO_RAW_PATH"
  "$NCSU_RAW_PATH_1"
  "$NCSU_RAW_PATH_2"
  "$NCSU_RAW_PATH_3"
)

RAW_LABELS=(
  "JUNO"
  "longterm_images2"
  "longterm_images"
  "GROW_DATA"
)

# Developed (JPG/metadata) locations
DEV_ENDPOINTS=(
  "$JUNO_EP"
  "$NCSU_1_EP"
  "$NCSU_2_EP"
  "$NCSU_3_EP"
)

DEV_PATHS=(
  "$JUNO_DEV_PATH"
  "$NCSU_DEV_PATH_1"
  "$NCSU_DEV_PATH_2"
  "$NCSU_DEV_PATH_3"
)

DEV_LABELS=(
  "JUNO"
  "longterm_images2"
  "longterm_images"
  "GROW_DATA"
)

# For dedup of per-location lists later (optional)
LOCATION_LABELS=("JUNO" "longterm_images2" "longterm_images" "GROW_DATA")

# Temp dir for per-location CSV fragments
TMP_DIR=$(mktemp -d)
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# -------------------------
# Main CSV header
# kind = upload | developed
# -------------------------
echo "batch_id,kind,source,raw_number,jpg_number,metadata_json_number" > "$OUTPUT_CSV"

# ======================================================
# 1) UPLOAD LOCATIONS (RAWs only) – PARALLELIZED
# ======================================================

for i in "${!RAW_ENDPOINTS[@]}"; do
  ep="${RAW_ENDPOINTS[$i]}"
  root="${RAW_PATHS[$i]}"
  src_label="${RAW_LABELS[$i]}"

  upload_batch_log="$LOGS_DIR/upload_batches_${src_label}_${TIMESTAMP}.txt"
  upload_batch_list="$TMP_DIR/upload_batches_${src_label}.txt"
  upload_csv_fragment="$TMP_DIR/upload_${src_label}.csv"

  echo "Listing UPLOAD batches in ${ep}:${root} (label=${src_label}) ..."
  # One big ls, then local work
  globus ls "${ep}:${root}" 2>/dev/null > "$upload_batch_list" || true
  # Clean up and normalize
  sed -e 's:/*$::' -e '/^$/d' "$upload_batch_list" | sort -u > "${upload_batch_list}.clean"
  mv "${upload_batch_list}.clean" "$upload_batch_list"

  num_batches=$(wc -l < "$upload_batch_list" || echo 0)
  echo "Found ${num_batches} UPLOAD batches in ${src_label}"

  # Save per-location batch list into logs (deduped)
  cp "$upload_batch_list" "$upload_batch_log"

  # Parallel per-batch processing
  if [[ "$num_batches" -gt 0 ]]; then
    cat "$upload_batch_list" \
    | xargs -P "$N_JOBS" -I{} bash -c '
        batch="$1"
        ep="$2"
        root="$3"
        src_label="$4"

        # Count RAWs in this batch
        raw_number=$(
          ( globus ls "${ep}:${root}/${batch}" 2>/dev/null || true ) \
          | grep -c "\.RAW$" || true
        )

        printf "%s,upload,%s,%d,0,0\n" "$batch" "$src_label" "$raw_number"
      ' _ {} "$ep" "$root" "$src_label" \
      > "$upload_csv_fragment"
  else
    : > "$upload_csv_fragment"
  fi
done

# ======================================================
# 2) DEVELOPED LOCATIONS (JPGs + metadata) – PARALLELIZED
# ======================================================

for i in "${!DEV_ENDPOINTS[@]}"; do
  ep="${DEV_ENDPOINTS[$i]}"
  root="${DEV_PATHS[$i]}"
  src_label="${DEV_LABELS[$i]}"

  developed_batch_log="$LOGS_DIR/developed_batches_${src_label}_${TIMESTAMP}.txt"
  developed_batch_list="$TMP_DIR/developed_batches_${src_label}.txt"
  developed_csv_fragment="$TMP_DIR/developed_${src_label}.csv"

  echo "Listing DEVELOPED batches in ${ep}:${root} (label=${src_label}) ..."
  globus ls "${ep}:${root}" 2>/dev/null > "$developed_batch_list" || true
  sed -e 's:/*$::' -e '/^$/d' "$developed_batch_list" | sort -u > "${developed_batch_list}.clean"
  mv "${developed_batch_list}.clean" "$developed_batch_list"

  num_batches=$(wc -l < "$developed_batch_list" || echo 0)
  echo "Found ${num_batches} DEVELOPED batches in ${src_label}"

  # Save per-location batch list into logs
  cp "$developed_batch_list" "$developed_batch_log"

  if [[ "$num_batches" -gt 0 ]]; then
    cat "$developed_batch_list" \
    | xargs -P "$N_JOBS" -I{} bash -c '
        batch="$1"
        ep="$2"
        root="$3"
        src_label="$4"

        jpg_number=$(
          ( globus ls "${ep}:${root}/${batch}/images" 2>/dev/null || true ) \
          | grep -c "\.jpg$" || true
        )

        metadata_json_number=$(
          ( globus ls "${ep}:${root}/${batch}/metadata" 2>/dev/null || true ) \
          | grep -c "\.json$" || true
        )

        printf "%s,developed,%s,0,%d,%d\n" "$batch" "$src_label" "$jpg_number" "$metadata_json_number"
      ' _ {} "$ep" "$root" "$src_label" \
      > "$developed_csv_fragment"
  else
    : > "$developed_csv_fragment"
  fi
done

# ======================================================
# 3) MERGE ALL CSV FRAGMENTS INTO MAIN CSV
# ======================================================

cat "$TMP_DIR"/upload_*.csv "$TMP_DIR"/developed_*.csv >> "$OUTPUT_CSV" 2>/dev/null || true

# (Optional) Dedup per-location logs
for label in "${LOCATION_LABELS[@]}"; do
  up_file="$LOGS_DIR/upload_batches_${label}_${TIMESTAMP}.txt"
  dev_file="$LOGS_DIR/developed_batches_${label}_${TIMESTAMP}.txt"

  if [[ -f "$up_file" ]]; then
    sort -u "$up_file" -o "$up_file"
  fi
  if [[ -f "$dev_file" ]]; then
    sort -u "$dev_file" -o "$dev_file"
  fi
done

echo "Done."
echo "Main CSV: $OUTPUT_CSV"
echo "Per-location batch lists: $LOGS_DIR/upload_batches_*_${TIMESTAMP}.txt and developed_batches_*_${TIMESTAMP}.txt"
