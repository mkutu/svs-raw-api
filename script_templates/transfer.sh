batch_id=$1
# -----------------------------
# 1. Load and activate environment
# -----------------------------
cd /tmp
module load miniconda
source activate /project/dash_agir/matthew.kutugata/software/miniforge3/envs/semif_prep
# Clone svs repo
git clone https://github.com/mkutu/svs-raw-api.git
cd svs-raw-api
# Install svs package
pip install -e .
bash scripts/validate_rawtherapee.sh
# -----------------------------
# 2. Vars
# -----------------------------
JUNO_EP="904c2108-90cf-11e8-9672-0a6d4e044368"
CERES_EP="f45a24f8-09ba-11ec-b342-1feaf93e3729"

# batch_id must come from sbatch --export=batch_id=... or set above
# e.g.: sbatch --export=batch_id=TX_2024-07-03 myscript.sh
if [ -z "$batch_id" ]; then
    echo "ERROR: batch_id is not set."
    exit 1
fi

JUNO_SRC="/LTS/project/dash_agir/semifield-upload/$batch_id"
CERES_90DAY_DEST="/90daydata/dash_agir/data/semifield-upload/$batch_id"
mkdir -p $CERES_90DAY_DEST

TMP_DEVELOPED_IMAGES_DIR="$TMPDIR/data/semifield-developed-images/$batch_id"
CERES_DEVELOPED_IMAGES_DIR="/project/dash_agir/matthew.kutugata/semifield-developed-images"

TMP_UPLOAD_IMAGE_DIR="$TMPDIR/data/semifield-upload"
mkdir -p $TMP_UPLOAD_IMAGE_DIR

mkdir -p "$TMP_DEVELOPED_IMAGES_DIR"
mkdir -p "$CERES_DEVELOPED_IMAGES_DIR"

# -----------------------------
# 3. Globus Transfer
# -----------------------------
# ✨ IMPORTANT: Globus CLI only works on **login nodes**, not compute nodes
# This will fail if run on a compute node.
#
# Instead, create a transfer task *before starting the compute job*.
# -----------------------------

echo $JUNO_SRC
echo $CERES_90DAY_DEST

echo "Attempting to run Globus transfer (WARNING: this usually fails on compute nodes)"
globus transfer "$JUNO_EP:$JUNO_SRC" "$CERES_EP:$CERES_90DAY_DEST" \
    --recursive --notify off

cp $CERES_90DAY_DEST $TMP_UPLOAD_IMAGE_DIR -r

echo "Done."

