import os
from pathlib import Path
import numpy as np
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from svs_raw_api import SVSRaw2DNG, HEIGHT, WIDTH

# -----------------------------
# CONFIG
# -----------------------------
rt_cli = '/home/mkutuga/SemiF-Preprocessing/squashfs-root/usr/bin/rawtherapee-cli'
matrix_path = Path("data/profiles/MD_calibration_matrix_optimized.npy")
config_path = Path("examples/svs_tags.yaml")
pp3_file = Path("data/profiles/MD_shr661_raw16.pp3")

svs_dng = SVSRaw2DNG(config_path, matrix_path)
tags = svs_dng.define_tags()

max_threads = 48          # total threads on node
processes = 12            # number of parallel workers
threads_per_process = max(1, max_threads // processes)


def load_raw_image(raw_path: Path) -> np.ndarray:
    raw = np.fromfile(raw_path, dtype=np.uint16)
    return raw.reshape((HEIGHT, WIDTH))


# -----------------------------
# WORKER FUNCTION
# -----------------------------
def process_one(raw_file: Path, out_dng_dir: Path, out_jpg_dir: Path):
    try:
        # Load RAW
        raw16 = load_raw_image(raw_file)

        # Write DNG
        dng_path = out_dng_dir / raw_file.stem
        svs_dng.run(tags, raw_file, raw16, dng_path)

        # Convert DNG → JPG using RawTherapee
        jpg_path = out_jpg_dir / f"{raw_file.stem}.jpg"

        cmd = [
            rt_cli,
            "-O", str(jpg_path),
            "-p", str(pp3_file),
            "-j100",
            "-js3",
            "-c", f"{dng_path}.dng"
        ]

        # Control OpenMP thread usage to avoid oversubscription
        env = {
            **os.environ,
            "OMP_NUM_THREADS": str(threads_per_process),
            "OMP_DYNAMIC": "TRUE",
            "OMP_NESTED": "FALSE"
        }

        subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)

        return (raw_file, "ok")

    except Exception as e:
        return (raw_file, f"ERROR: {e}")


# -----------------------------
# MAIN
# -----------------------------
batch_id = "MD_2025-10-22"
input_dir = Path(f"/mnt/research-projects/s/screberg/longterm_images2/semifield-upload/{batch_id}/")
raw_files = sorted(input_dir.glob("*.RAW"))

dng_out = Path(f"data/dngs/{batch_id}")
jpg_out = Path(f"data/jpgs/{batch_id}")
dng_out.mkdir(parents=True, exist_ok=True)
jpg_out.mkdir(parents=True, exist_ok=True)

# Run in parallel
with ProcessPoolExecutor(max_workers=processes) as ex:
    futures = [
        ex.submit(process_one, rf, dng_out, jpg_out)
        for rf in raw_files
    ]

    for fut in as_completed(futures):
        raw_path, status = fut.result()
        print(raw_path.name, status)
