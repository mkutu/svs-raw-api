#!/usr/bin/env bash

#SBATCH --job-name=check_batches
#SBATCH -A dash_agir
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH -o "/project/dash_agir/matthew.kutugata/logs/check_batches-%j.out"
#SBATCH -e "/project/dash_agir/matthew.kutugata/logs/check_batches-%j.err"

set -euo pipefail

# --- CONFIG ---
JUNO_EP="904c2108-90cf-11e8-9672-0a6d4e044368"


UPLOAD_ROOT="/LTS/project/dash_agir/semifield-upload"
DEV_ROOT="/LTS/project/dash_agir/semifield-developed-images"

OUTPUT_CSV="batch_summary.csv"
LOC_DEV_LIST="developed_batches.txt"
LOC_UPLOAD_LIST="upload_batches.txt"

# --- Helper: safe temp files ---
UPLOAD_LIST=$(mktemp)
DEV_LIST=$(mktemp)
ALL_BATCHES=$(mktemp)
UPLOAD_TMP=$(mktemp)
DEV_TMP=$(mktemp)
META_TMP=$(mktemp)

cleanup() {
    rm -f "$UPLOAD_LIST" "$DEV_LIST" "$ALL_BATCHES" "$UPLOAD_TMP" "$DEV_TMP"
}
trap cleanup EXIT

echo "Listing upload batches from $UPLOAD_ROOT ..."
globus ls "$JUNO_EP:$UPLOAD_ROOT" > "$UPLOAD_LIST"
cp "$UPLOAD_LIST" "$LOC_UPLOAD_LIST"

echo "Listing developed batches from $DEV_ROOT ..."
globus ls "$JUNO_EP:$DEV_ROOT" > "$DEV_LIST"
cp "$DEV_LIST" "$LOC_DEV_LIST"

# Unique batch IDs making sure not to lose any
cat "$UPLOAD_LIST" "$DEV_LIST" | sort | uniq > "$ALL_BATCHES"

echo "Writing summary to $OUTPUT_CSV"
echo "batch_id,upload_exists,developed_exists,raw_number,jpg_number,metadata_exists,metadata_json_number" > "$OUTPUT_CSV"

while read -r batch; do
    [[ -z "$batch" ]] && continue

    upload_exists=0
    developed_exists=0
    raw_number=0
    jpg_number=0
    metadata_exists=0
    metadata_json_number=0

    # ---- Check upload dir & count RAWs ----
    if globus ls "$JUNO_EP:$UPLOAD_ROOT/$batch" > "$UPLOAD_TMP" 2>/dev/null; then
        upload_exists=1
        # Count .RAW (case-sensitive; tweak to '\.[Rr][Aa][Ww]$' if needed)
        raw_number=$(grep -c '\.RAW$' "$UPLOAD_TMP" || true)
    fi

    # ---- Check developed/images dir & count JPGs ----
    if globus ls "$JUNO_EP:$DEV_ROOT/$batch/images" > "$DEV_TMP" 2>/dev/null; then
        developed_exists=1
        # Count .jpg (case-sensitive; tweak to '\.[Jj][Pp][Ee]?[Gg]$' if mixed)
        jpg_number=$(grep -c '\.jpg$' "$DEV_TMP" || true)
    fi

    # ---- Check developed/metadata dir & count JSONs ----
    if globus ls "$JUNO_EP:$DEV_ROOT/$batch/metadata" > "$META_TMP" 2>/dev/null; then
        metadata_exists=1
        # Count .json files
        metadata_json_number=$(grep -c '\.json$' "$META_TMP" || true)
    fi

    echo "$batch,$upload_exists,$developed_exists,$raw_number,$jpg_number,$metadata_exists,$metadata_json_number" >> "$OUTPUT_CSV"
done < "$ALL_BATCHES"

echo "Done. Summary written to: $OUTPUT_CSV"
