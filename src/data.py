# Standard library
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

# Third-party
import numpy as np

# Local application imports
from image_processing_api.constants import (
    BIT_DEPTH,
    BLACK_LEVEL_SHIFTED,
    CAMERA_MODEL,
    COLORCHECKER_REFERENCE_SRGB,
    EFFECTIVE_RANGE,
    F_NUMBER,
    F_NUMBER_RANGE,
    FOCAL_LENGTH_MM,
    HEIGHT,
    IMAGE_CIRCLE_MM,
    LENS_MODEL,
    NUMBER_OF_PATCHES,
    RAW_MAX_VALUE,
    SENSOR_MODEL,
    WIDTH,
)

@dataclass
class CameraConfig:
    """Camera configuration parameters."""
    width: int = WIDTH
    height: int = HEIGHT
    bit_depth: int = BIT_DEPTH
    sensor_model: str = SENSOR_MODEL
    camera_model: str = CAMERA_MODEL
    lens_model: str = LENS_MODEL
    lens_focal_length: float = FOCAL_LENGTH_MM  # mm
    lens_f_number: float = F_NUMBER
    lens_f_range: Tuple[float, float] = F_NUMBER_RANGE
    lens_image_circle: float = IMAGE_CIRCLE_MM  # mm


@dataclass
class ProcessingConfig:
    """Parameters for image processing pipeline."""    
    exposure_stops: float = 0.3 # Exposure
    
    # Demosaicing
    demosaic_method: str = 'AHD'  # 'AHD', 'VNG', etc.
    # Tone mapping
    tone_mapping: str = 'rolloff'  # 'rolloff', 'reinhard', 'aces', 'reinhard_extended', 'none'
    
    # Tone mapping specific parameters
    highlight_threshold: float = 0.85  # For rolloff
    highlight_smoothness: float = 0.10  # For rolloff
    white_point: float = 4.0  # For reinhard_extended
    
    # Output options
    output_dir: Path = Path('./processed_images')
    output_use_param_prefix: bool = False
    output_fname_prefix: str = None
    output_format: str = 'jpg'  # 'jpg' or 'png'
    output_quality: int = 100  # For JPEG (1-100)
    output_preview: bool = True
    output_fullres: bool = True
    preview_scale: float = 0.25

    # Output encoding
    gamma: float = 0.85
    output_bit_depth: int = 8  # 8 or 16

    # Path to calculated color matrix (if any)
    color_matrix_path: Optional[Path] = None
    color_matrix: Optional[np.ndarray] = None

    # Load color matrix from file if path is provided
    def __post_init__(self):
        if self.color_matrix_path is not None:
            try:
                self.color_matrix = np.load(self.color_matrix_path)
            except Exception as e:
                raise ValueError(f"Failed to load color matrix from {self.color_matrix_path}: {e}")

    def __setattr__(self, name, value):
        """
        Intercept assignment to color_matrix_path so assigning after init
        (e.g. config.color_matrix_path = "path/to/matrix.npy") will auto-load.
        """
        if name == "color_matrix_path":
            # Normalize to Path or None
            path_val = Path(value) if value is not None else None
            object.__setattr__(self, name, path_val)
            if path_val is not None:
                try:
                    mat = np.load(path_val)
                    object.__setattr__(self, "color_matrix", mat)
                except Exception as e:
                    raise ValueError(f"Failed to load color matrix from {path_val}: {e}")
            else:
                object.__setattr__(self, "color_matrix", None)
        else:
            object.__setattr__(self, name, value)


@dataclass
class CalibrationConfig:
    """Configuration for ColorChecker calibration."""
    colorchecker_raw_path: Path = None
    output_dir: Path = Path('./calibration_results')
    checker_top_left: Tuple[int, int] = None
    checker_bottom_right: Tuple[int, int] = None
    adjust_white: bool = False
    exclude_white: bool = False
    display_scale: Optional[float] = None
    calc_wb: bool = False
    num_patches: int = NUMBER_OF_PATCHES
    height: int = HEIGHT
    width: int = WIDTH
    black_level_shifted: int = BLACK_LEVEL_SHIFTED  # 16-bit value
    raw_max_value: int = RAW_MAX_VALUE # 4095 << 4 (12-bit left-shifted to 16-bit)
    effective_range: int = EFFECTIVE_RANGE  # 65520 - 368

    # create the output directory if it doesn't exist
    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class CalibrationResult:
    """Results from ColorChecker calibration."""
    color_matrix: np.ndarray
    wb_gains: Dict[str, float]
    measured_colors: np.ndarray
    corrected_colors: np.ndarray
    reference_colors: np.ndarray
    patch_coords: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    mean_error_before: float
    mean_error_after: float
    max_error_before: float
    max_error_after: float
    clipped_patches: List[int]
    timestamp: str
    output_dir: Path = field(default_factory=lambda: Path('./calibration_results/data'))

    def export_calibration_results(self, prefix: str = "calibration"):
        """Export all calibration results."""
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save color matrix
        matrix_path = output_dir / f"{prefix}_matrix_{self.timestamp}.npy"
        np.save(matrix_path, self.color_matrix)
        
        # Save calibration data
        data_path = output_dir / f"{prefix}_data_{self.timestamp}.npy"
        np.save(data_path, {
            'measured_colors': self.measured_colors,
            'corrected_colors': self.corrected_colors,
            'reference_colors': self.reference_colors,
            'patch_coords': self.patch_coords,
            'clipped_patches': self.clipped_patches
        })
        # Save calibration data as json
        json_data_path = output_dir / f"{prefix}_data_{self.timestamp}.json"
        with open(json_data_path, 'w') as f:
            json.dump({
                'measured_colors': self.measured_colors.tolist(),
                'corrected_colors': self.corrected_colors.tolist(),
                'reference_colors': self.reference_colors.tolist(),
                'patch_coords': self.patch_coords,
                'clipped_patches': self.clipped_patches
            }, f, indent=2)
        
        # Save JSON metadata
        json_path = output_dir / f"{prefix}_metadata_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'timestamp': self.timestamp,
                'wb_gains': self.wb_gains,
                'mean_error_before': float(self.mean_error_before),
                'mean_error_after': float(self.mean_error_after),
                'max_error_before': float(self.max_error_before),
                'max_error_after': float(self.max_error_after),
                'clipped_patches': self.clipped_patches
            }, f, indent=2)
        
        return {
            'matrix': matrix_path,
            'data': data_path,
            'json': json_path
        }

@dataclass
class ProcessingResult:
    """Results from processing an image."""
    input_path: str
    output_path: str
    preview_path: Optional[str]
    status: str  # 'success' or 'error'
    error: Optional[str] = None
    
    def __repr__(self):
        if self.status == 'success':
            return f"ProcessingResult(✓ {Path(self.output_path).name})"
        else:
            return f"ProcessingResult(✗ {Path(self.input_path).name}: {self.error})"

@dataclass
class BatchResult:
    """Results from batch processing."""
    total: int
    successful: int
    failed: int
    results: List[ProcessingResult]
    output_dir: str
    
    def print_summary(self):
        """Print a nice summary."""
        print("\n" + "="*70)
        print("BATCH PROCESSING COMPLETE")
        print("="*70)
        print(f"Total: {self.total}")
        print(f"Successful: {self.successful}")
        print(f"Failed: {self.failed}")
        print(f"Output directory: {self.output_dir}")
        print("="*70)


    
    def save(self, output_dir: Path, prefix: str = "calibration"):
        """Save all calibration results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save color matrix
        matrix_path = output_dir / f"{prefix}_matrix.npy"
        np.save(matrix_path, self.color_matrix)
        
        # Save calibration data
        data_path = output_dir / f"{prefix}_data_{timestamp}.npy"
        np.save(data_path, {
            'measured_colors': self.measured_colors,
            'corrected_colors': self.corrected_colors,
            'reference_colors': self.reference_colors,
            'patch_coords': self.patch_coords,
            'clipped_patches': self.clipped_patches
        })
        
        # Save JSON metadata
        json_path = output_dir / f"{prefix}_data_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'timestamp': self.timestamp,
                'wb_gains': self.wb_gains,
                'mean_error_before': float(self.mean_error_before),
                'mean_error_after': float(self.mean_error_after),
                'max_error_before': float(self.max_error_before),
                'max_error_after': float(self.max_error_after),
                'clipped_patches': self.clipped_patches
            }, f, indent=2)
        
        return {
            'matrix': matrix_path,
            'data': data_path,
            'json': json_path
        }
    
@dataclass
class RawTherapeeConfig:
    """Configuration for RawTherapee processing."""
    rt_cli_path: Path = Path('./path/to/rawtherapee-cli')
    rt_pp3_path: Path = None  # Path to .pp3 profile
    temp_dir: Path = Path('./rawtherapee_processing')
    use_temp_dir: bool = True
    max_threads: int = 30  # Max threads for RawTherapee
    num_instances: int = 12  # Number of parallel instances
    LANG: str = "en_US.UTF-8" # Language setting for RawTherapee
    OMP_DYNAMIC: bool = True
    OMP_NESTED: bool = False  # Whether to set OMP_NESTED env var

