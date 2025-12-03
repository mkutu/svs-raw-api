"""
Snakemake script: Convert RAW to DNG
Called by Snakemake rule 'raw_to_dng'
"""
import sys
from pathlib import Path
import numpy as np
import logging

# Setup logging
logging.basicConfig(
    filename=str(snakemake.log[0]),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Add svs_raw_api to path
    repo_root = Path(snakemake.params.repo_root)
    sys.path.insert(0, str(repo_root / "src"))

    from svs_raw_api import SVSRaw2DNG

    # Get parameters
    raw_path = Path(snakemake.input.raw)
    dng_path = Path(snakemake.output.dng)
    svs_tags = Path(snakemake.params.svs_tags)
    color_matrix = Path(snakemake.params.color_matrix)
    height = snakemake.params.height
    width = snakemake.params.width

    logger.info(f"Processing {raw_path.name}")
    logger.info(f"Output: {dng_path}")

    # Load RAW image
    logger.info("Loading RAW image...")
    raw_image = np.fromfile(raw_path, dtype=np.uint16).astype(np.uint16)
    raw_image_16 = np.reshape(raw_image, (height, width))
    logger.info(f"Loaded image: {width}x{height} pixels")

    # Initialize DNG converter
    logger.info("Initializing DNG converter...")
    svs_dng = SVSRaw2DNG(svs_tags, color_matrix)
    tags = svs_dng.define_tags()

    # Create output directory
    dng_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to DNG
    logger.info("Converting to DNG...")
    dng_path_base = dng_path.with_suffix('')
    svs_dng.run(tags, raw_path, raw_image_16, dng_path_base)

    # Verify output
    if dng_path.exists():
        size_mb = dng_path.stat().st_size / (1024**2)
        logger.info(f"SUCCESS: Created {dng_path.name} ({size_mb:.2f} MB)")
    else:
        raise FileNotFoundError(f"DNG file was not created: {dng_path}")

except Exception as e:
    logger.error(f"FAILED: {str(e)}", exc_info=True)
    raise
