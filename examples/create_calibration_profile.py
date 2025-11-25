#!/usr/bin/env python3
"""
Example Usage: Before and After Optimization

This script shows how to use both the original and optimized versions.
"""

from pathlib import Path
import numpy as np

from svs_raw_api import (
    CalibrationConfig, CalibrationResult, ImageProcessor
)
# ============================================================================
# SETUP (Same for both versions)
# ============================================================================

# Load calibration matrix
raw_colorchecker        = Path('data/raw/MD_1759501672.RAW')
outoput_path              = Path('data/processed/MD_calibration_matrix_optimized.npy')

# ============================================================================
# CONFIGURATION (Same for both versions)
# ============================================================================

processor = ImageProcessor()

calib_config = CalibrationConfig()
calib_config.colorchecker_raw_path  = raw_colorchecker
calib_config.output_path            = outoput_path
calib_config.checker_top_left       = (5258, 5863)
calib_config.checker_bottom_right   = (6043, 6817)
calib_config.adjust_white           = False
calib_config.exclude_white          = False
calib_config.display_scale          = 0.5
calib_config.calc_wb                = False

calib_result: CalibrationResult = processor.calibrate(calib_config)
calib_result.export_calibration_results()