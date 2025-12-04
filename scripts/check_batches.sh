#!/usr/bin/env bash
#SBATCH --job-name=check_batches
#SBATCH --partition=short
#SBATCH --account=dash_agir
#SBATCH --time=04:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=/project/dash_agir/logs/check_batches_%x_%j.out.log
#SBATCH --error=/project/dash_agir/logs/check_batches_%x_%j.err.log

set -euo pipefail

# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
JUNO_EP="904c2108-90cf-11e8-9672-0a6d4e044368"
NCSU_1_EP="2f7f6170-8d5c-11e9-8e6a-029d279f7e24"
NCSU_2_EP="2f7f6170-8d5c-11e9-8e6a-029d279f7e24"
NCSU_3_EP="2f7f6170-8d5c-11e9-8e6a-029d279f7e24"

#----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
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
LOGS_RAW_DIR="/project/dash_agir/logs/raw"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_CSV="$LOGS_DIR/batch_summary_${TIMESTAMP}.csv"

mkdir -p "$LOGS_DIR"
mkdir -p "$LOGS_RAW_DIR"
# ----------------------------------------------------------------------
# Arrays for multi-location handling
# ----------------------------------------------------------------------

# Upload (RAW) locations
RAW_ENDPOINTS=(
  "$NCSU_1_EP"
  "$NCSU_2_EP"
  "$NCSU_3_EP"
  "$JUNO_EP"
)

RAW_PATHS=(
  "$NCSU_RAW_PATH_1"
  "$NCSU_RAW_PATH_2"
  "$NCSU_RAW_PATH_3"
  "$JUNO_RAW_PATH"
)

# Developed (JPG/metadata) locations
DEV_ENDPOINTS=(
  "$NCSU_1_EP"
  "$NCSU_2_EP"
  "$NCSU_3_EP"
  "$JUNO_EP"
)

DEV_PATHS=(
  "$NCSU_DEV_PATH_1"
  "$NCSU_DEV_PATH_2"
  "$NCSU_DEV_PATH_3"
  "$JUNO_DEV_PATH"
)

# Labels we expect, for cleanup later
LOCATION_LABELS=("longterm_images2" "longterm_images" "GROW_DATA" "JUNO")

# ----------------------------------------------------------------------
# Helper: map root path → logical source label
# ----------------------------------------------------------------------
get_source_label() {
    local root="$1"
    if [[ "$root" == *"longterm_images2"* ]]; then
        echo "longterm_images2"
    elif [[ "$root" == *"/longterm_images/"* ]]; then
        echo "longterm_images"
    elif [[ "$root" == *"GROW_DATA"* ]]; then
        echo "GROW_DATA"
    else
        echo "JUNO"
    fi
}

# ----------------------------------------------------------------------
# CSV header
# kind = upload | developed
# source = JUNO | longterm_images2 | longterm_images | GROW_DATA
# ----------------------------------------------------------------------
echo "batch_id,kind,source,raw_number,jpg_number,metadata_json_number" > "$OUTPUT_CSV"

# ----------------------------------------------------------------------
# Scan UPLOAD locations (RAWs only)
# ----------------------------------------------------------------------
for i in "${!RAW_ENDPOINTS[@]}"; do
    ep="${RAW_ENDPOINTS[$i]}"
    root="${RAW_PATHS[$i]}"
    # echo "Root: $root"
    src_label=$(get_source_label "$root")
    # echo "Source label: $src_label"

    upload_batch_log="$LOGS_RAW_DIR/upload_batches_${src_label}_${TIMESTAMP}.txt"

    echo "Scanning UPLOAD batches in ${ep}:${root} ..."

    ( globus ls "${ep}:${root}" 2>/dev/null || true ) | while read -r batch; do
        [[ -z "$batch" ]] && continue
        batch=${batch%/}

        # Count .RAW files in this batch (in the upload folder)
        raw_number=$(
            ( globus ls "${ep}:${root}/${batch}" 2>/dev/null || true ) \
            | grep -c '\.RAW$' || true
        )

        # Append to per-location batch list
        echo "$batch" >> "$upload_batch_log"

        # Write row to CSV
        echo "$batch,upload,$src_label,$raw_number,0,0" >> "$OUTPUT_CSV"
    done
done

# ----------------------------------------------------------------------
# Scan DEVELOPED locations (JPGs + metadata JSONs only)
# ----------------------------------------------------------------------
for i in "${!DEV_ENDPOINTS[@]}"; do
    ep="${DEV_ENDPOINTS[$i]}"
    root="${DEV_PATHS[$i]}"
    src_label=$(get_source_label "$root")

    developed_batch_log="$LOGS_RAW_DIR/developed_batches_${src_label}_${TIMESTAMP}.txt"

    echo "Scanning DEVELOPED batches in ${ep}:${root} ..."

    ( globus ls "${ep}:${root}" 2>/dev/null || true ) | while read -r batch; do
        [[ -z "$batch" ]] && continue
        batch=${batch%/}

        # Count JPGs in images subfolder
        jpg_number=$(
            ( globus ls "${ep}:${root}/${batch}/images" 2>/dev/null || true ) \
            | grep -c '\.jpg$' || true
        )

        # Count JSONs in metadata subfolder
        metadata_json_number=$(
            ( globus ls "${ep}:${root}/${batch}/metadata" 2>/dev/null || true ) \
            | grep -c '\.json$' || true
        )

        # Append to per-location batch list
        echo "$batch" >> "$developed_batch_log"

        # Write row to CSV
        echo "$batch,developed,$src_label,0,$jpg_number,$metadata_json_number" >> "$OUTPUT_CSV"
    done
done

# ----------------------------------------------------------------------
# Deduplicate per-location batch lists
# ----------------------------------------------------------------------
for label in "${LOCATION_LABELS[@]}"; do
    up_file="$LOGS_RAW_DIR/upload_batches_${label}_${TIMESTAMP}.txt"
    dev_file="$LOGS_RAW_DIR/developed_batches_${label}_${TIMESTAMP}.txt"

    if [[ -f "$up_file" ]]; then
        sort -u "$up_file" -o "$up_file"
    fi

    if [[ -f "$dev_file" ]]; then
        sort -u "$dev_file" -o "$dev_file"
    fi
done

echo "Done. Summary written to: $OUTPUT_CSV"
echo "Per-location batch lists written to: $LOGS_DIR/upload_batches_*_${TIMESTAMP}.txt and $LOGS_DIR/developed_batches_*_${TIMESTAMP}.txt"
