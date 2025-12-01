import os
from pathlib import Path
import numpy as np
import subprocess

from svs_raw_api import SVSRaw2DNG, HEIGHT, WIDTH

def load_raw_image(raw_path: Path) -> np.ndarray:
    raw_image = np.fromfile(raw_path, dtype=np.uint16).astype(np.uint16)
    raw_image_16 = np.reshape(raw_image, (HEIGHT, WIDTH))
    return raw_image_16

# --- Load the raw 16-bit image directly ---
# raw_file = Path("data/raw/MD_1759501672.RAW")
# raw_file = Path("/mnt/research-projects/s/screberg/longterm_images2/semifield-upload/MD_2025-10-09/MD_1760033880.RAW")
# raw_file = Path("/mnt/research-projects/s/screberg/longterm_images2/semifield-upload/MD_2025-10-03/MD_1759502864.RAW")
raw_file = Path("/mnt/research-projects/s/screberg/longterm_images2/semifield-upload/NC_2025-06-23/NC_1750697912.RAW")
# --- Load your custom color matrix (3x3) ---
matrix_path = Path("data/profiles/MD_calibration_matrix_optimized.npy")

# --- PP3 file for JPG conversion ---
pp3_file = Path("data/profiles/MD_shr661_raw16.pp3")

# --- Create DNG Tags ---
config_path = Path("examples/svs_tags.yaml")

# --- LOAD RAW IMAGE ---
batch_id = "NC_2025-06-23"
lts_dir = Path(f"/mnt/research-projects/s/screberg/longterm_images2/semifield-upload/{batch_id}/")
# raw_images = sorted(list(lts_dir.glob("*.RAW")))
raw_images = [raw_file]

# --- Output path ---
output_dir = Path(f"data/dngs/{batch_id}")
output_dir.mkdir(parents=True, exist_ok=True)

# jpg_output_dir = Path(f"data/jpgs/{batch_id}")
# jpg_output_dir.mkdir(parents=True, exist_ok=True)
jpg_output_dir = Path("/mnt/research-projects/s/screberg/longterm_images2/semifield-developed-images/NC_2025-06-23/images/")

# --- CLI ---
rt_cli='/home/mkutuga/SemiF-Preprocessing/squashfs-root/usr/bin/rawtherapee-cli'

# --- WRITE DNG ---
svs_dng = SVSRaw2DNG(config_path, matrix_path)
t = svs_dng.define_tags()
for raw_file in raw_images:
    # raw_image_16 = load_raw_image(raw_file)
    # dng_path = output_dir / f"{raw_file.stem}"
    # svs_dng.run(t, raw_file, raw_image_16, dng_path)
    dng_path = "/home/mkutuga/svs-raw-api/data/dngs/NC_2025-06-23/NC_1750697912"

    jpg_output_path = jpg_output_dir / f"{raw_file.stem}.jpg"

    cmd = [
            rt_cli,
            "-O", jpg_output_path,
            "-p", pp3_file,
            "-j100",
            "-js3",
            "-Y",
            "-c", f"{dng_path}.dng"
        ]
    try:
        max_threads = 50  # Total number of threads to use
        num_instances = 12  # Expected number of parallel threads
        threads_per_instance = max(1, max_threads // num_instances) #
        # prevents oversubscription of resources

        # Set environment per process
        env = {
            **os.environ,
            "LANG": "en_US.UTF-8",
            "OMP_NUM_THREADS": str(threads_per_instance),
            "OMP_DYNAMIC": "TRUE",  # Allows OpenMP to optimize thread count
            "OMP_NESTED": "FALSE"  # Disables nested parallelism
        }
        _ = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=env
        )
    except subprocess.CalledProcessError as e:
        print(f"Error converting dng to jpg: {e}")
        continue