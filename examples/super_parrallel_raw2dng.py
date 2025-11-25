import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from svs_raw_api import SVSRaw2DNG, HEIGHT, WIDTH

# -----------------------------
# CONFIG
# -----------------------------
BATCH_ID = "MD_2025-10-16"

INPUT_DIR = Path(
    f"/mnt/research-projects/s/screberg/longterm_images2/semifield-upload/{BATCH_ID}/"
)

DNG_OUT_DIR = Path(f"data/dngs/{BATCH_ID}")
JPG_OUT_DIR = Path(f"data/jpgs/{BATCH_ID}")
DNG_OUT_DIR.mkdir(parents=True, exist_ok=True)
JPG_OUT_DIR.mkdir(parents=True, exist_ok=True)

MATRIX_PATH = Path("data/profiles/MD_calibration_matrix_optimized.npy")
CONFIG_PATH = Path("examples/svs_tags.yaml")
PP3_FILE = Path("data/profiles/MD_shr661_raw16.pp3")
RT_CLI = "/home/mkutuga/SemiF-Preprocessing/squashfs-root/usr/bin/rawtherapee-cli"

# Tune based on box
N_DNG_WORKERS = 4          # RAW -> DNG
N_JPG_WORKERS = 6          # DNG -> JPG
MAX_OMP_THREADS = 48       # total logical cores
OMP_THREADS_PER_RT = max(1, MAX_OMP_THREADS // N_JPG_WORKERS)


def load_raw_image(raw_path: Path) -> np.ndarray:
    arr = np.fromfile(raw_path, dtype=np.uint16)
    return arr.reshape((HEIGHT, WIDTH))


# -----------------------------
# WORKER: RAW -> DNG
# -----------------------------
def raw_to_dng(raw_file: Path,
               config_path: Path,
               matrix_path: Path,
               dng_out_dir: Path) -> Path:
    """
    Convert a single RAW file to DNG and return the DNG path.
    NOTE: instantiate SVSRaw2DNG inside worker (not pickled).
    """
    raw_file = Path(raw_file)
    dng_out_dir = Path(dng_out_dir)

    svs_dng = SVSRaw2DNG(config_path, matrix_path)
    tags = svs_dng.define_tags()

    raw16 = load_raw_image(raw_file)
    dng_path = dng_out_dir / raw_file.stem

    svs_dng.run(tags, raw_file, raw16, dng_path)
    return dng_path  # without .dng extension if that's how your run() works


# -----------------------------
# WORKER: DNG -> JPG
# -----------------------------
def dng_to_jpg(dng_path: Path,
               jpg_out_dir: Path,
               rt_cli: str,
               pp3_file: Path,
               threads_per_rt: int) -> Path:
    """
    Convert a single DNG file to JPG with RawTherapee and return JPG path.
    """
    dng_path = Path(dng_path)
    jpg_out_dir = Path(jpg_out_dir)
    jpg_path = jpg_out_dir / f"{dng_path.stem}.jpg"

    base_env = dict(os.environ)
    base_env.update({
        "LANG": "en_US.UTF-8",
        "OMP_DYNAMIC": "TRUE",
        "OMP_NESTED": "FALSE",
        "OMP_NUM_THREADS": str(threads_per_rt),
    })

    cmd = [
        rt_cli,
        "-O", str(jpg_path),
        "-p", str(pp3_file),
        "-j100",
        "-js3",
        "-c", f"{dng_path}.dng",
    ]

    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env=base_env,
    )

    return jpg_path


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    raw_files = sorted(INPUT_DIR.glob("*.RAW"))
    print(f"Found {len(raw_files)} RAW files in {INPUT_DIR}")
    if not raw_files:
        return

    jpg_futures = []

    # Two executors: one for RAW->DNG, one for DNG->JPG
    with ProcessPoolExecutor(max_workers=N_DNG_WORKERS) as dng_executor, \
         ProcessPoolExecutor(max_workers=N_JPG_WORKERS) as jpg_executor:

        # Submit all RAW -> DNG tasks
        dng_futures = {
            dng_executor.submit(
                raw_to_dng,
                rf,
                CONFIG_PATH,
                MATRIX_PATH,
                DNG_OUT_DIR,
            ): rf
            for rf in raw_files
        }

        # As each DNG finishes, immediately submit DNG -> JPG
        for fut in as_completed(dng_futures):
            raw_path = dng_futures[fut]
            try:
                dng_path = fut.result()
                print(f"[DNG DONE] {raw_path.name} -> {dng_path.name}")

                jpg_fut = jpg_executor.submit(
                    dng_to_jpg,
                    dng_path,
                    JPG_OUT_DIR,
                    RT_CLI,
                    PP3_FILE,
                    OMP_THREADS_PER_RT,
                )
                jpg_futures.append((jpg_fut, dng_path))

            except Exception as e:
                print(f"[DNG ERROR] {raw_path}: {e}")

        # Wait for all JPG conversions
        for jpg_fut, dng_path in jpg_futures:
            try:
                jpg_path = jpg_fut.result()
                print(f"[JPG DONE] {dng_path.name} -> {jpg_path.name}")
            except Exception as e:
                print(f"[JPG ERROR] {dng_path}: {e}")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
