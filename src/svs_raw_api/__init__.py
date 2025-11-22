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
    WIDTH,
    HEIGHT,
    BLACK_LEVEL_SHIFTED,
    BLACK_LEVEL_12BIT,
    RAW_MAX_VALUE,
    EFFECTIVE_RANGE,
    PATCH_NAMES,
    NUMBER_OF_PATCHES,
    COLORCHECKER_REFERENCE_SRGB
)

# Data structures
from .data import (
    CameraConfig,
    ProcessingConfig,
    CalibrationConfig,
    CalibrationResult,
    ProcessingResult,
    BatchResult,
    RawTherapeeConfig
)

# Image processing functions
from .processing_utils import (
    load_raw_image,
    demosaic_image,
    apply_color_correction,
    apply_gamma_correction,
    apply_exposure_compensation,
    apply_highlight_rolloff,
    apply_tone_curve,
    check_clipping_stats
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
    'WIDTH',
    'HEIGHT',
    'BLACK_LEVEL_SHIFTED',
    'BLACK_LEVEL_12BIT',
    'RAW_MAX_VALUE',
    'EFFECTIVE_RANGE',
    'PATCH_NAMES',
    'NUMBER_OF_PATCHES',
    'COLORCHECKER_REFERENCE_SRGB',
    
    # Data structures
    'CameraConfig',
    'ProcessingConfig',
    'CalibrationConfig',
    'CalibrationResult',
    'ProcessingResult',
    'BatchResult',
    'RawTherapeeConfig',
    
    # Image processing
    'load_raw_image',
    'demosaic_image',
    'apply_color_correction',
    'apply_gamma_correction',
    'apply_exposure_compensation',
    'apply_highlight_rolloff',
    'apply_tone_curve',
    'apply_tone_curve_with_prescale',
    'check_clipping_stats',
    
    # ColorChecker
    'isolate_colorchecker',
    'MultiPatchSelector',
    'extract_patch_colors',
    'diagnose_patch_clipping',
    'save_patch_visualization',
    'save_comparison_image',
    'analyze_color_matrix',

    # Parameter grids
    'PARAMETER_GRIDS',
    'get_baseline_params',
    
    # Main API
    'ImageProcessor',
]