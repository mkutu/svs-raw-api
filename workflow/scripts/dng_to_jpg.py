"""
Snakemake script: Convert DNG to JPG using RawTherapee
Called by Snakemake rule 'dng_to_jpg'
"""
import subprocess
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    filename=str(snakemake.log[0]),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Get parameters
    dng_path = Path(snakemake.input.dng)
    jpg_path = Path(snakemake.output.jpg)
    pp3_profile = Path(snakemake.params.pp3_profile)
    rawtherapee_cli = Path(snakemake.params.rawtherapee_cli)
    threads = snakemake.params.threads

    logger.info(f"Converting {dng_path.name} to JPG")
    logger.info(f"Profile: {pp3_profile.name}")
    logger.info(f"Threads: {threads}")

    # Verify inputs exist
    if not dng_path.exists():
        raise FileNotFoundError(f"DNG not found: {dng_path}")
    if not pp3_profile.exists():
        raise FileNotFoundError(f"PP3 profile not found: {pp3_profile}")
    if not rawtherapee_cli.exists():
        raise FileNotFoundError(f"RawTherapee CLI not found: {rawtherapee_cli}")

    # Create output directory
    jpg_path.parent.mkdir(parents=True, exist_ok=True)

    # Build RawTherapee command
    cmd = [
        str(rawtherapee_cli),
        "-O", str(jpg_path),
        "-p", str(pp3_profile),
        "-j100",      # JPG quality 100
        "-js3",       # JPG subsampling 4:4:4 (best quality)
        "-Y",         # Overwrite output
        "-c", str(dng_path)
    ]

    # Setup environment for OpenMP
    env = {
        **os.environ,
        "LANG": "en_US.UTF-8",
        "OMP_NUM_THREADS": str(threads),
        "OMP_DYNAMIC": "TRUE",
        "OMP_NESTED": "FALSE"
    }

    logger.info(f"Running: {' '.join(cmd[:3])} ...")

    # Run conversion
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=600  # 10 minute timeout
    )

    # Log stdout if verbose
    if result.stdout:
        logger.debug(f"RawTherapee output:\n{result.stdout}")

    # Verify output
    if jpg_path.exists():
        size_mb = jpg_path.stat().st_size / (1024**2)
        logger.info(f"SUCCESS: Created {jpg_path.name} ({size_mb:.2f} MB)")
    else:
        raise FileNotFoundError(f"JPG file was not created: {jpg_path}")

except subprocess.CalledProcessError as e:
    logger.error(f"RawTherapee failed: {e.stderr}")
    raise
except subprocess.TimeoutExpired:
    logger.error(f"Conversion timed out after 10 minutes")
    raise
except Exception as e:
    logger.error(f"FAILED: {str(e)}", exc_info=True)
    raise
