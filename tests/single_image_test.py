"""
CLI for processing SVS RAW images on HPC
Usage: python -m svs_raw_api.cli process --input <path> --output <path>
"""
import argparse
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import yaml

from svs_raw_api import SVSRaw2DNG, HEIGHT, WIDTH


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file with environment variable expansion."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Expand ${var} references in paths
    def expand_vars(obj):
        if isinstance(obj, dict):
            return {k: expand_vars(v) for k, v in obj.items()}
        elif isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj
    
    return expand_vars(config)


def setup_logging(log_dir: Path, job_id: str = None) -> logging.Logger:
    """Setup logging with timestamp."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_suffix = f"_{job_id}" if job_id else ""
    log_file = log_dir / f"process_{timestamp}{job_suffix}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return logging.getLogger(__name__)


def load_raw_image(raw_path: Path, height: int, width: int) -> np.ndarray:
    """Load and reshape raw image."""
    raw_image = np.fromfile(raw_path, dtype=np.uint16).astype(np.uint16)
    raw_image_16 = np.reshape(raw_image, (height, width))
    return raw_image_16


def convert_dng_to_jpg(dng_path: Path, jpg_path: Path, pp3_file: Path, 
                       rt_cli: Path, threads: int, logger: logging.Logger):
    """Convert DNG to JPG using RawTherapee."""
    cmd = [
        str(rt_cli),
        "-O", str(jpg_path),
        "-p", str(pp3_file),
        "-j100",
        "-js3",
        "-Y",
        "-c", str(dng_path)
    ]
    
    env = {
        **os.environ,
        "LANG": "en_US.UTF-8",
        "OMP_NUM_THREADS": str(threads),
        "OMP_DYNAMIC": "TRUE",
        "OMP_NESTED": "FALSE"
    }
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=300  # 5 minute timeout
        )
        logger.info(f"Converted {dng_path.name} to JPG")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert {dng_path.name}: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout converting {dng_path.name}")
        return False


def process_single_image(raw_file: Path, output_dir: Path, config: dict, logger: logging.Logger):
    """Process a single RAW file to DNG and JPG."""
    logger.info(f"Processing {raw_file.name}")
    
    # Load configuration
    paths = config['paths']
    processing = config['processing']
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize DNG converter
    svs_dng = SVSRaw2DNG(
        Path(paths['svs_tags']),
        Path(paths['color_matrix'])
    )
    tags = svs_dng.define_tags()
    
    # Load and convert to DNG
    raw_image = load_raw_image(raw_file, HEIGHT, WIDTH)
    dng_path = output_dir / raw_file.stem
    
    svs_dng.run(tags, raw_file, raw_image, dng_path)
    dng_file = dng_path.with_suffix('.dng')
    
    if not dng_file.exists():
        logger.error(f"Failed to create DNG: {dng_file}")
        return False
    
    logger.info(f"Created DNG: {dng_file}")
    
    # Convert to JPG
    jpg_path = output_dir / f"{raw_file.stem}.jpg"
    success = convert_dng_to_jpg(
        dng_file,
        jpg_path,
        Path(paths['pp3_profile']),
        Path(paths['rawtherapee_cli']),
        processing['threads_per_image'],
        logger
    )
    
    return success


def process_batch(input_dir: Path, output_dir: Path, config: dict, 
                  pattern: str = "*.RAW", logger: logging.Logger = None):
    """Process all RAW files in a directory."""
    raw_files = sorted(input_dir.glob(pattern))
    
    if not raw_files:
        logger.warning(f"No files matching {pattern} in {input_dir}")
        return
    
    logger.info(f"Found {len(raw_files)} files to process")
    
    success_count = 0
    for i, raw_file in enumerate(raw_files, 1):
        logger.info(f"[{i}/{len(raw_files)}] Processing {raw_file.name}")
        if process_single_image(raw_file, output_dir, config, logger):
            success_count += 1
    
    logger.info(f"Completed: {success_count}/{len(raw_files)} successful")


def main():
    parser = argparse.ArgumentParser(description="Process SVS RAW images")
    parser.add_argument('--config', type=Path, 
                       default=Path('~/repos/svs-raw-api/config/scinet.yaml').expanduser(),
                       help='Path to config file')
    parser.add_argument('--input', type=Path, required=True,
                       help='Input directory or single RAW file')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory')
    parser.add_argument('--pattern', default='*.RAW',
                       help='File pattern to match (default: *.RAW)')
    parser.add_argument('--threads', type=int,
                       help='Threads per image (overrides config)')
    parser.add_argument('--job-id', 
                       help='SLURM job ID for logging')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override threads if specified
    if args.threads:
        config['processing']['threads_per_image'] = args.threads
    
    # Setup logging
    log_dir = Path(config['paths']['logs_dir'])
    logger = setup_logging(log_dir, args.job_id)
    
    logger.info(f"Config loaded from {args.config}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    
    # Process
    if args.input.is_file():
        process_single_image(args.input, args.output, config, logger)
    elif args.input.is_dir():
        process_batch(args.input, args.output, config, args.pattern, logger)
    else:
        logger.error(f"Input path does not exist: {args.input}")
        return 1
    
    logger.info("Processing complete")
    return 0


if __name__ == "__main__":
    exit(main())