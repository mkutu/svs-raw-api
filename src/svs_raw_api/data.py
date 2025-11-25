# Standard library
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

# Third-party
import numpy as np

# Local application imports
from svs_raw_api.constants import (
    BLACK_LEVEL_SHIFTED,
    EFFECTIVE_RANGE,
    HEIGHT,
    NUMBER_OF_PATCHES,
    RAW_MAX_VALUE,
    WIDTH,
)



@dataclass
class CalibrationConfig:
    """Configuration for ColorChecker calibration."""
    colorchecker_raw_path: Path = None
    output_dir: Path = Path('./calibration_results')
    output_path: Path = None
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
    color_matrix: np.ndarray = None
    forward_matrix: np.ndarray = None
    wb_gains: np.ndarray = None
    ccm_rational: List[List[int]] = None
    fm_rational: List[List[int]] = None
    as_shot_neutral: List[List[int]] = None

    # create the output directory if it doesn't exist
    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_npy(cls, path: Path, matrix_den: int) -> "CalibrationConfig":
        data = np.load(path, allow_pickle=True).item()

        cm = data["color_matrix"].T
        fm = data["forward_matrix"].T
        wb = np.asarray(data["wb_gains"], dtype=float)

        # Convert matrices to DNG rational form
        ccm_r = [[int(round(v * matrix_den)), matrix_den] for v in cm.reshape(-1)]
        fm_r = [[int(round(v * matrix_den)), matrix_den] for v in fm.reshape(-1)]

        # AsShotNeutral = inverse of WB gains
        r_gain, g_gain, b_gain = wb
        asn = [
            [int(round(matrix_den / r_gain)), matrix_den],
            [int(round(matrix_den / g_gain)), matrix_den],
            [int(round(matrix_den / b_gain)), matrix_den],
        ]

        return cls(
            color_matrix=cm,
            forward_matrix=fm,
            wb_gains=wb,
            ccm_rational=ccm_r,
            fm_rational=fm_r,
            as_shot_neutral=asn,
        )

@dataclass
class CalibrationResult:
    """Results from ColorChecker calibration."""
    color_matrix: np.ndarray
    forward_matrix: np.ndarray
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
    output_dir: Path = None
    output_path: Path = None

    def export_calibration_results(self):
        """Export all calibration results."""
        if self.output_path is not None:
            output_dir = self.output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = self.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save color matrix
        matrix_path = self.output_path if self.output_path is not None else output_dir / f"calibration_matrix_{self.timestamp}.npy"
        matrix_data = {
            'color_matrix': self.color_matrix,
            'forward_matrix': self.forward_matrix,
            'wb_gains': self.wb_gains
        }
        np.save(matrix_path, matrix_data)
        
        # Save calibration data
        data_path = self.output_path.parent / f"calibration_data_{self.timestamp}.npy" if self.output_path is not None else output_dir / f"calibration_data_{self.timestamp}.npy"
        np.save(data_path, {
            'measured_colors': self.measured_colors,
            'corrected_colors': self.corrected_colors,
            'reference_colors': self.reference_colors,
            'patch_coords': self.patch_coords,
            'clipped_patches': self.clipped_patches
        })
        # Save calibration data as json
        json_data_path = self.output_path.parent / f"calibration_data_{self.timestamp}.json" if self.output_path is not None else output_dir / f"calibration_data_{self.timestamp}.json"
        with open(json_data_path, 'w') as f:
            json.dump({
                'measured_colors': self.measured_colors.tolist(),
                'corrected_colors': self.corrected_colors.tolist(),
                'reference_colors': self.reference_colors.tolist(),
                'patch_coords': self.patch_coords,
                'clipped_patches': self.clipped_patches,
                'timestamp': self.timestamp,
                'mean_error_before': float(self.mean_error_before),
                'mean_error_after': float(self.mean_error_after),
                'max_error_before': float(self.max_error_before),
                'max_error_after': float(self.max_error_after),
                'clipped_patches': self.clipped_patches
            }, f, indent=2)
        
        return {
            'matrix': matrix_path,
            'data': data_path,
            'json': json_data_path
        }

