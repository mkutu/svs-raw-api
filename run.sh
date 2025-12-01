#!/bin/bash
#SBATCH --job-name=color_correction
#SBATCH -A dash_agir
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 04:00:00
#SBATCH -o color_correction-%j.out

batch_id=$1  # Get the batch ID from the first argument
# Load and activate conda/mamba module
module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh 
MAMBA_ENV_PATH=$(mamba env list | grep 'agcv$' | awk '{print $NF}')
source activate $MAMBA_ENV_PATH

# Setting up directories for data transfer
# Use Globus to transfer data (if available)
JUNO_EP="904c2108-90cf-11e8-9672-0a6d4e044368"
CERES_EP="f45a24f8-09ba-11ec-b342-1feaf93e3729"

# LTS to 90DAYDATA transfer paths
JUNO_SRC="/project/dash_agir/semifield-upload/$batch_id"
CERES_90DAY_DEST="/90daydata/dash_agir/data/semfield-upload/$batch_id"

TMP_DEVELOPED_IMAGES_DIR=$TMPDIR/data/semifield-developed-images/$batch_id

CERES_DEVELOPED_IMAGES_DIR="/project/dash_agir/matthew.kutugata/semifield-developed-images/$batch_id"

mkdir -p $TMP_DEVELOPED_IMAGES_DIR
mkdir -p $CERES_DEVELOPED_IMAGES_DIR


globus transfer $JUNO_EP:$JUNO_SRC $CERES_EP:$CERES_90DAY_DEST --recursive --notify off

/bin/cp -r $CERES_90DAY_DEST $TMP_DEVELOPED_IMAGES_DIR

# Function to copy data back before exiting
cleanup() {
    echo "Job ending - copying data back from $TMP_DEVELOPED_IMAGES_DIR..."
    rsync -avh $TMP_DEVELOPED_IMAGES_DIR $CERES_DEVELOPED_IMAGES_DIR
    echo "Cleanup complete!"
}

# Register the cleanup function to run when job ends
trap cleanup EXIT

pip install -e .
