"""
Image Processing Pipeline for Agricultural Computer Vision

A modular pipeline for processing RAW images from industrial cameras,
with a focus on ColorChecker-based calibration, color correction,
and consistent image quality for computer vision applications.

Camera: SVS-Vistek shr661CXGE with Sony IMX661LQA sensor
Lens: inspec.x L 4/60 (60mm, f/4.0)
"""

# Version
__version__ = "1.0.0"

# Core configuration
from .constants import (
    SENSOR_MODEL,
    CAMERA_MODEL,
    LENS_MODEL,
    FOCAL_LENGTH_MM,
    IMAGE_CIRCLE_MM,
    F_NUMBER,
    F_NUMBER_RANGE,
    BIT_DEPTH,
    BIT_DEPTH_SHIFTED,
    WIDTH,
    HEIGHT,
    BLACK_LEVEL_SHIFTED,
    BLACK_LEVEL_12BIT,
    WHITE_LEVEL_SHIFTED,
    RAW_MAX_VALUE,
    EFFECTIVE_RANGE,
    PATCH_NAMES,
    NUMBER_OF_PATCHES,
    COLORCHECKER_REFERENCE_SRGB,
    M_SRGB_TO_XYZ,
)

# Data structures
from .data import (
    CalibrationConfig,
    CalibrationResult
)

# Image processing functions
from .processing_utils import (
    load_raw_image,
    demosaic_image
)


# ColorChecker selection and extraction
from .selection import (
    isolate_colorchecker,
    MultiPatchSelector,
    extract_patch_colors,
    diagnose_patch_clipping,
    save_patch_visualization,
    save_comparison_image,
    analyze_color_matrix
)

from .ccm import (
    srgb_to_linear,
    srgb_to_xyz_d65,
    load_calibration_json,
    compute_forward_matrix,
    compute_color_matrix, 
    format_for_dng,
    compute_error_stats,
    compute_wb,
)

from .dng_tags import (
    SVCamTagConfig,
    ColorConfig
)

from .raw2dng import SVSRaw2DNG
# Main API
from .pipeline import ImageProcessor

__all__ = [
    # Version
    '__version__',
    
    # Constants
    'SENSOR_MODEL',
    'CAMERA_MODEL',
    'LENS_MODEL',
    'FOCAL_LENGTH_MM',
    'IMAGE_CIRCLE_MM',
    'F_NUMBER',
    'F_NUMBER_RANGE',
    'BIT_DEPTH',
    'BIT_DEPTH_SHIFTED',
    'WIDTH',
    'HEIGHT',
    'BLACK_LEVEL_SHIFTED',
    'BLACK_LEVEL_12BIT',
    'WHITE_LEVEL_SHIFTED',
    'RAW_MAX_VALUE',
    'EFFECTIVE_RANGE',
    'PATCH_NAMES',
    'NUMBER_OF_PATCHES',
    'COLORCHECKER_REFERENCE_SRGB',
    'M_SRGB_TO_XYZ',
    
    # Data structures
    'CalibrationConfig',
    'CalibrationResult',
    
    # Image processing
    'load_raw_image',
    'demosaic_image',
    
    # ColorChecker
    'isolate_colorchecker',
    'MultiPatchSelector',
    'extract_patch_colors',
    'diagnose_patch_clipping',
    'save_patch_visualization',
    'save_comparison_image',
    'analyze_color_matrix',

    # CCM utilities
    'srgb_to_linear',
    'srgb_to_xyz_d65',
    'load_calibration_json',
    'compute_forward_matrix',
    'compute_color_matrix',
    'format_for_dng',
    'compute_error_stats',
    'compute_wb',

    # Parameter grids
    'PARAMETER_GRIDS',
    'get_baseline_params',

    # DNG tags
    'SVCamTagConfig',
    'ColorConfig',
    
    # Raw to DNG conversion
    'SVSRaw2DNG',
    
    # Main API
    'ImageProcessor',

]