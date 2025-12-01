import os
import json
import csv
from datetime import datetime
import logging
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional
import numpy as np
import pandas as pd
import yaml

from svs_raw_api import SVSRaw2DNG, HEIGHT, WIDTH

# -----------------------------
# CONFIGURATION LOADING
# -----------------------------
class Config:
    """Configuration container loaded from YAML file."""
    
    def __init__(self, config_path: str = "reprocess_config.yaml"):
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please create a config file based on reprocess_config.yaml.example"
            )
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Paths
        paths = config['paths']
        self.summary_json_path = Path(paths['summary_json'])
        self.upload_root = Path(paths['upload_root'])
        self.developed_roots_by_lts = {
            k: Path(v) for k, v in paths['developed_roots'].items()
        }
        self.dng_root = Path(paths['dng_root'])
        self.jpg_lts_root = Path(paths['jpg_lts_root'])
        self.log_dir = Path(paths['log_dir'])
        self.log_csv_path = Path(paths['log_csv'])
        
        # Calibration
        calib = config['calibration']
        self.matrix_path = Path(calib['matrix_path'])
        self.config_path = Path(calib['config_path'])
        self.pp3_file = Path(calib['pp3_file'])
        self.rt_cli = calib['rt_cli']
        
        # Processing
        proc = config['processing']
        self.n_dng_workers = proc['n_dng_workers']
        self.n_jpg_workers = proc['n_jpg_workers']
        self.max_omp_threads = proc['max_omp_threads']
        
        # Calculate OMP threads per RT instance if not specified
        omp_per_rt = proc.get('omp_threads_per_rt')
        if omp_per_rt is None:
            self.omp_threads_per_rt = max(1, self.max_omp_threads // self.n_jpg_workers)
        else:
            self.omp_threads_per_rt = omp_per_rt
        
        # Create necessary directories
        self.dng_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> None:
        """Validate critical paths and settings."""
        errors = []
        
        # Check critical files exist
        if not self.summary_json_path.exists():
            errors.append(f"Summary JSON not found: {self.summary_json_path}")
        if not self.matrix_path.exists():
            errors.append(f"Calibration matrix not found: {self.matrix_path}")
        if not self.config_path.exists():
            errors.append(f"SVS config not found: {self.config_path}")
        if not self.pp3_file.exists():
            errors.append(f"PP3 profile not found: {self.pp3_file}")
        if not Path(self.rt_cli).exists():
            errors.append(f"RawTherapee CLI not found: {self.rt_cli}")
        
        # Check directories
        if not self.upload_root.exists():
            errors.append(f"Upload root directory not found: {self.upload_root}")
        
        if errors:
            raise ValueError(
                "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )


# -----------------------------
# GLOBAL CONFIG AND LOGGING
# -----------------------------
# Config will be loaded in main()
cfg: Optional[Config] = None
logger = logging.getLogger("reprocess_raw2jpg")

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

    # File handler with timestamp
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = cfg.log_dir / f"reprocess_raw2jpg_{run_ts}.log"
    
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Configuration loaded from: reprocess_config.yaml")
    return logger


def load_raw_image(raw_path: Path) -> np.ndarray:
    """Load RAW image from file."""
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

    with open(cfg.log_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writerow(row)
    logger.debug(
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

    logger.debug(
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

    upload_raw_dir = cfg.upload_root / batch_name

    dev_root = cfg.developed_roots_by_lts.get(lts)
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
    dng_out_dir = cfg.dng_root / batch_name
    jpg_out_dir = cfg.jpg_lts_root / batch_name / "images"
    dng_out_dir.mkdir(parents=True, exist_ok=True)
    jpg_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[START] {state} {batch_name} (LTS={lts}): "
        f"{len(raw_files)} RAWs -> {dng_out_dir} / {jpg_out_dir}"
    )

    jpg_futures = []

    # Two executors: one for RAW->DNG, one for DNG->JPG
    with ProcessPoolExecutor(max_workers=cfg.n_dng_workers) as dng_executor, \
         ProcessPoolExecutor(max_workers=cfg.n_jpg_workers) as jpg_executor:

        # Submit all RAW -> DNG tasks
        dng_futures = {
            dng_executor.submit(
                raw_to_dng,
                rf,
                cfg.config_path,
                cfg.matrix_path,
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
                    cfg.rt_cli,
                    cfg.pp3_file,
                    cfg.omp_threads_per_rt,
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
                dng_file = Path(f"{dng_path}.dng")
                if dng_file.exists():
                    os.remove(dng_file)
            except Exception as e:
                logger.exception(
                    f"[JPG ERROR] {batch_name} :: {dng_path}: {e}"
                )

    logger.info(f"[DONE] {state} {batch_name}: pipeline complete.")


# -----------------------------
# MAIN PIPELINE (ALL BATCHES)
# -----------------------------
def main():
    global cfg
    
    # Load configuration first
    try:
        cfg = Config("reprocess_config.yaml")
        cfg.validate()
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        return 1
    
    # Init logging after config is loaded
    setup_logging()
    
    # Log configuration summary
    logger.info("=" * 80)
    logger.info("BATCH REPROCESSING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"DNG workers: {cfg.n_dng_workers}")
    logger.info(f"JPG workers: {cfg.n_jpg_workers}")
    logger.info(f"OMP threads per RT: {cfg.omp_threads_per_rt}")
    logger.info("=" * 80)

    # Ensure log CSV exists with header
    init_log_csv(cfg.log_csv_path)

    # Load batches
    batches = load_batches_from_summary(cfg.summary_json_path)
    logger.info(f"Loaded {len(batches)} batch records from {cfg.summary_json_path}")

    # Filter out upload_only and already processed batches
    df = pd.DataFrame(batches)
    
    # Data that is currently being processed
    done = df[df['upload_only'] == False]
    done = done[done['has_upload']]
    target_batches = df[~df["name"].isin(done["name"])]
    
    logger.info(f"{len(target_batches)} batches have upload_only==False.")
    logger.info(
        f"{len(target_batches)} batches have developed images (has_upload==True)."
    )
    
    # Process each batch
    for _, batch in target_batches.iterrows():
        process_batch(batch)

    logger.info("=" * 80)
    logger.info("All eligible batches processed.")
    logger.info(f"Processing log: {cfg.log_csv_path}")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())