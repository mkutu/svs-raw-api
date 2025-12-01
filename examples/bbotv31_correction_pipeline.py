import os
import json
import csv
from datetime import datetime
import logging
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from svs_raw_api import SVSRaw2DNG, HEIGHT, WIDTH

# -----------------------------
# CONFIG
# -----------------------------
# JSON summary from the batch/date-range script
SUMMARY_JSON_PATH = Path("examples/bbotv31_batches.json")

# Upload RAWs always live here (by your earlier setup)
UPLOAD_ROOT = Path(
    "/mnt/research-projects/s/screberg/longterm_images2/semifield-upload"
)

# Developed roots, keyed by the "lts" tag stored in the JSON
DEVELOPED_ROOTS_BY_LTS = {
    "longterm_images": Path(
        "/mnt/research-projects/s/screberg/longterm_images/semifield-developed-images"
    ),
    "GROW_DATA": Path(
        "/mnt/research-projects/s/screberg/GROW_DATA/semifield-developed-images"
    ),
    "longterm_images2": Path(
        "/mnt/research-projects/s/screberg/longterm_images2/semifield-developed-images"
    ),
}

# Local output roots (per-batch subfolders will be created)
DNG_ROOT = Path("data/dngs")
JPG_LTS_ROOT = Path(
    "/mnt/research-projects/s/screberg/longterm_images2/semifield-developed-images"
)
DNG_ROOT.mkdir(parents=True, exist_ok=True)

# NOTE: These are MD-specific paths in your original script.
# If you later process NC/TX with different calibration, you can branch on state.
MATRIX_PATH = Path("data/profiles/MD_calibration_matrix_optimized.npy")
CONFIG_PATH = Path("examples/svs_tags.yaml")
PP3_FILE = Path("data/profiles/MD_shr661_raw16.pp3")
RT_CLI = "/home/mkutuga/svs-raw-api/squashfs-root/usr/bin/rawtherapee-cli"

# Tune based on box
N_DNG_WORKERS = 4          # RAW -> DNG
N_JPG_WORKERS = 6          # DNG -> JPG
MAX_OMP_THREADS = 48       # total logical cores
OMP_THREADS_PER_RT = max(1, MAX_OMP_THREADS // N_JPG_WORKERS)

LOG_CSV_PATH = Path("reprocessed_images_log.csv")
LOG_FIELDS = [
    "processed_at",
    "state",
    "batch",
    "batch_date",
    "image_stem",
    "raw_filename",
    "raw_path",
    "dng_path",
    "jpg_path",
    "lts",
    "seasons",
    "upload_only",
    "jpg_mtime",
]

# -----------------------------
# PYTHON LOGGING CONFIG
# -----------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"reprocess_raw2jpg_{RUN_TS}.log"

# global logger object
logger = logging.getLogger("reprocess_raw2jpg")


def setup_logging() -> logging.Logger:
    """
    Configure root logger with:
      - FileHandler (timestamped filename)
      - StreamHandler (stdout)
      using format: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s
    """
    if logger.handlers:
        # Already configured
        return logger

    logger.setLevel(logging.INFO)

    fmt = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    formatter = logging.Formatter(fmt)

    # File handler
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging initialized. Log file: {LOG_FILE}")
    return logger


def load_raw_image(raw_path: Path) -> np.ndarray:
    arr = np.fromfile(raw_path, dtype=np.uint16)
    return arr.reshape((HEIGHT, WIDTH))


# -----------------------------
# CSV LOGGING HELPERS
# -----------------------------
def init_log_csv(path: Path) -> None:
    """Create the CSV with header if it does not exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
    logger.info(f"Initialized CSV log at {path}")


def log_reprocessed_image(
    batch: dict,
    raw_path: Path,
    dng_path: Path,
    jpg_path: Path,
) -> None:
    """
    Append a single row to the CSV for a successfully reprocessed image.
    """
    processed_at = datetime.now().isoformat(timespec="seconds")

    state = batch.get("state", "NA")
    batch_name = batch["name"]
    batch_date = batch.get("date", "")
    lts = batch.get("lts", "UNKNOWN")
    seasons_list = batch.get("seasons", [])
    seasons_str = "|".join(sorted(seasons_list)) if seasons_list else ""
    upload_only = batch.get("upload_only", False)

    jpg_mtime = jpg_path.stat().st_mtime if jpg_path.exists() else None

    row = {
        "processed_at": processed_at,
        "state": state,
        "batch": batch_name,
        "batch_date": batch_date,
        "image_stem": raw_path.stem,
        "raw_filename": raw_path.name,
        "raw_path": str(raw_path),
        "dng_path": str(dng_path),
        "jpg_path": str(jpg_path),
        "lts": lts,
        "seasons": seasons_str,
        "upload_only": upload_only,
        "jpg_mtime": jpg_mtime,
    }

    with open(LOG_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writerow(row)
    logger.info(
        f"Logged reprocessed image: state={state}, batch={batch_name}, "
        f"image={raw_path.name}"
    )


# -----------------------------
# WORKER: RAW -> DNG
# -----------------------------
def raw_to_dng(
    raw_file: Path,
    config_path: Path,
    matrix_path: Path,
    dng_out_dir: Path,
) -> Path:
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

    logger.debug(f"Starting RAW->DNG for {raw_file}")
    svs_dng.run(tags, raw_file, raw16, dng_path)
    logger.debug(f"Completed RAW->DNG for {raw_file} -> {dng_path}")
    return dng_path  # without .dng extension if that's how your run() works


# -----------------------------
# WORKER: DNG -> JPG
# -----------------------------
def dng_to_jpg(
    dng_path: Path,
    jpg_out_dir: Path,
    rt_cli: str,
    pp3_file: Path,
    threads_per_rt: int,
) -> Path:
    """
    Convert a single DNG file to JPG with RawTherapee and return JPG path.
    """
    dng_path = Path(dng_path)
    jpg_out_dir = Path(jpg_out_dir)
    jpg_path = jpg_out_dir / f"{dng_path.stem}.jpg"

    logger.info(
        f"[RT] Converting {dng_path} -> {jpg_path} with {threads_per_rt} OMP threads"
    )

    base_env = dict(os.environ)
    base_env.update(
        {
            "LANG": "en_US.UTF-8",
            "OMP_DYNAMIC": "TRUE",
            "OMP_NESTED": "FALSE",
            "OMP_NUM_THREADS": str(threads_per_rt),
        }
    )

    cmd = [
        rt_cli,
        "-O",
        str(jpg_path),
        "-p",
        str(pp3_file),
        "-j100",
        "-js3",
        "-Y",
        "-c",
        f"{dng_path}.dng",
    ]

    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env=base_env,
    )

    logger.debug(f"Completed DNG->JPG for {dng_path} -> {jpg_path}")
    return jpg_path


# -----------------------------
# JSON HELPERS
# -----------------------------
def load_batches_from_summary(summary_path: Path) -> list[dict]:
    """
    Flatten the JSON summary into a unique list of batch dicts:
    Each entry is expected to have:
      - name
      - date
      - image_count
      - upload_only
      - lts
      - seasons (list)
    """
    logger.info(f"Loading batch summary from {summary_path}")
    with open(summary_path, "r") as f:
        summary = json.load(f)

    batches_by_key: dict[tuple[str, str], dict] = {}

    for state, s_info in summary["states"].items():
        # Seasons
        for season_info in s_info["seasons"].values():
            for b in season_info["batches"]:
                key = (state, b["name"])
                b_with_state = dict(b)
                b_with_state.setdefault("state", state)
                batches_by_key[key] = b_with_state

        # Unassigned
        for b in s_info["unassigned"]["batches"]:
            key = (state, b["name"])
            b_with_state = dict(b)
            b_with_state.setdefault("state", state)
            batches_by_key[key] = b_with_state

    batches = list(batches_by_key.values())
    logger.info(f"Loaded {len(batches)} unique batch records from summary.")
    return batches


def get_paths_for_batch(batch: dict) -> tuple[Path, Path | None]:
    """
    Given a batch record from the JSON summary, return:
      - upload_raw_dir: Path to the upload RAW directory
      - developed_images_dir: Path to the developed images/ directory (or None if missing)
    """
    batch_name = batch["name"]
    lts = batch.get("lts")

    upload_raw_dir = UPLOAD_ROOT / batch_name

    dev_root = DEVELOPED_ROOTS_BY_LTS.get(lts)
    if dev_root is None:
        logger.warning(
            f"No developed root found for batch={batch_name}, lts={lts}"
        )
        return upload_raw_dir, None

    developed_images_dir = dev_root / batch_name / "images"
    return upload_raw_dir, developed_images_dir


def select_raws_for_reprocessing(
    upload_raw_dir: Path,
    developed_images_dir: Path,
) -> list[Path]:
    """
    From the upload_raw_dir, select only RAW files whose stem exists
    in the developed images directory.
    This enforces: "only reprocess the images that are in the developed folder."
    """
    if not upload_raw_dir.exists():
        logger.warning(f"[SKIP] Upload RAW dir does not exist: {upload_raw_dir}")
        return []

    if not developed_images_dir.exists():
        logger.warning(
            f"[SKIP] Developed images dir does not exist: {developed_images_dir}"
        )
        return []

    # Stems of all developed images (any extension)
    dev_stems = {
        p.stem
        for p in developed_images_dir.iterdir()
        if p.is_file()
    }

    if not dev_stems:
        logger.warning(f"[SKIP] No developed images found in {developed_images_dir}")
        return []

    # RAW files in upload dir whose stem matches a developed image stem
    raw_files = sorted(upload_raw_dir.glob("*.RAW"))
    selected = [rf for rf in raw_files if rf.stem in dev_stems]

    logger.info(
        f"[INFO] {upload_raw_dir.name}: {len(raw_files)} RAW files, "
        f"{len(selected)} matched to developed images"
    )
    return selected


# -----------------------------
# PER-BATCH PIPELINE
# -----------------------------
def process_batch(batch: dict) -> None:
    """
    Reprocess a single batch:
      - Skip upload_only batches (no developed images).
      - Filter RAWs to only those that have a developed image.
      - Run RAW->DNG->JPG pipeline on that filtered set.
      - Log each successfully reprocessed image to CSV.
    """
    batch_name = batch["name"]
    state = batch.get("state", "NA")
    upload_only = batch.get("upload_only", False)
    lts = batch.get("lts", "UNKNOWN")

    if upload_only:
        logger.info(
            f"[SKIP] {state} {batch_name}: upload_only batch (no developed images)."
        )
        return

    upload_raw_dir, developed_images_dir = get_paths_for_batch(batch)

    if developed_images_dir is None:
        logger.info(
            f"[SKIP] {state} {batch_name}: no developed root found for lts={lts}."
        )
        return

    raw_files = select_raws_for_reprocessing(upload_raw_dir, developed_images_dir)
    if not raw_files:
        logger.info(f"[SKIP] {state} {batch_name}: no RAW files to reprocess.")
        return

    # Per-batch output dirs
    dng_out_dir = DNG_ROOT / batch_name
    jpg_out_dir = JPG_LTS_ROOT / batch_name / "images"
    dng_out_dir.mkdir(parents=True, exist_ok=True)
    jpg_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[START] {state} {batch_name} (LTS={lts}): "
        f"{len(raw_files)} RAWs -> {dng_out_dir} / {jpg_out_dir}"
    )

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
                dng_out_dir,
            ): rf
            for rf in raw_files
        }

        # As each DNG finishes, immediately submit DNG -> JPG
        for fut in as_completed(dng_futures):
            raw_path = dng_futures[fut]
            try:
                dng_path = fut.result()
                logger.info(
                    f"[DNG DONE] {batch_name} :: {raw_path.name} -> {dng_path.name}"
                )

                jpg_fut = jpg_executor.submit(
                    dng_to_jpg,
                    dng_path,
                    jpg_out_dir,
                    RT_CLI,
                    PP3_FILE,
                    OMP_THREADS_PER_RT,
                )
                # keep raw_path so we can log later
                jpg_futures.append((jpg_fut, dng_path, raw_path))

            except Exception as e:
                logger.exception(
                    f"[DNG ERROR] {batch_name} :: {raw_path}: {e}"
                )

        # Wait for all JPG conversions, log successes
        for jpg_fut, dng_path, raw_path in jpg_futures:
            try:
                jpg_path = jpg_fut.result()
                logger.info(
                    f"[JPG DONE] {batch_name} :: {dng_path.name} -> {jpg_path.name}"
                )
                # Log only on successful JPG creation
                log_reprocessed_image(batch, raw_path, dng_path, jpg_path)
                # remove DNG after successful JPG conversion
                os.remove(f"{dng_path}.dng")
            except Exception as e:
                logger.exception(
                    f"[JPG ERROR] {batch_name} :: {dng_path}: {e}"
                )

    logger.info(f"[DONE] {state} {batch_name}: pipeline complete.")


# -----------------------------
# MAIN PIPELINE (ALL BATCHES)
# -----------------------------
def main():
    # init logging first so everything else can log
    setup_logging()

    # Ensure log CSV exists with header
    init_log_csv(LOG_CSV_PATH)

    batches = load_batches_from_summary(SUMMARY_JSON_PATH)
    logger.info(f"Loaded {len(batches)} batch records from {SUMMARY_JSON_PATH}")

    # Filter out upload_only here as a first pass (extra guard)
    target_batches = [b for b in batches if not b.get("upload_only", False)]
    logger.info(f"{len(target_batches)} batches have upload_only==False.")
    # Filter out batches without where has_upload is False
    target_batches = [b for b in target_batches if b.get("has_upload", True)]
    logger.info(
        f"{len(target_batches)} batches have developed images (has_upload==True)."
    )
    batches_already_run = (
        "MD_2025-06-30",
    )
    for batch in target_batches:
        if batch["name"] in batches_already_run:
            logger.info(f"[SKIP] {batch['state']} {batch['name']}: already processed.")
            continue
        process_batch(batch)

    logger.info("All eligible batches processed.")


if __name__ == "__main__":
    main()
