#!/usr/bin/env python3
"""
Example Usage: Before and After Optimization

This script shows how to use both the original and optimized versions.
"""

from pathlib import Path
import numpy as np

from image_processing_api import (
    CalibrationConfig, CalibrationResult, ProcessingConfig
)
from image_processing_api import ImageProcessor as OptimizedProcessor
# ============================================================================
# SETUP (Same for both versions)
# ============================================================================

# Load calibration matrix
calibration_path = Path("calibration_results/data/calibration_matrix_20251119_171642.npy")
color_matrix = np.load(calibration_path)

# Paths
batch_id = 'MD_2025-10-03'
raw_dir = Path(f'/mnt/research-projects/s/screberg/longterm_images2/semifield-upload/{batch_id}')
# raw_dir = Path('./colorchecker_raw')
output_dir = Path("processed_images")

# ============================================================================
# CONFIGURATION (Same for both versions)
# ============================================================================


processor = OptimizedProcessor(n_workers=12)

calib_config = CalibrationConfig()
calib_config.colorchecker_raw_path  = Path('/mnt/research-projects/s/screberg/longterm_images2/semifield-upload/MD_2025-10-03/MD_1759501672.RAW')
calib_config.checker_top_left       = (5258, 5863)
calib_config.checker_bottom_right   = (6043, 6817)
calib_config.adjust_white           = False
calib_config.exclude_white          = False
calib_config.display_scale          = 0.5
calib_config.calc_wb                = False

calib_result: CalibrationResult = processor.calibrate(calib_config)
calib_result.export_calibration_results()

config = ProcessingConfig(
    color_matrix=color_matrix,
    exposure_stops=0.3,
    tone_mapping='rolloff',
    highlight_threshold=0.85,
    highlight_smoothness=0.10,
    gamma=0.85,
    output_dir=output_dir,
    output_format='jpg',
    output_quality=100,
    output_preview=True,
    output_fullres=True,
    preview_scale=0.25
)


# Create processor with 6 workers
processor_optimized = OptimizedProcessor(n_workers=12)
processor_optimized.load_calibration(calibration_path)

# Process batch (exact same API!)
result_optimized = processor_optimized.process_batch(
    input_dir=raw_dir,
    params=config,
    pattern="*.RAW",
    limit=None,
    use_parallel=True,   # Enable parallel processing
    show_progress=True   # Show progress bar
)