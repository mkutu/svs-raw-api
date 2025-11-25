"""
Agricultural Image Processing Pipeline API (OPTIMIZED)
For SVS-Vistek shr661CXGE camera with inspec.x L 4/60 lens

This module provides a clean API for RAW image processing with ColorChecker calibration.
OPTIMIZED VERSION with parallel processing support.
"""
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from datetime import datetime
from svs_raw_api.constants import COLORCHECKER_REFERENCE_SRGB

from svs_raw_api.data import (
    CalibrationConfig,
    CalibrationResult
)

from svs_raw_api.selection import MultiPatchSelector
from svs_raw_api.processing_utils import (
    load_raw_image, demosaic_image,
)
from svs_raw_api.ccm import (
    srgb_to_linear,
    srgb_to_xyz_d65,
    compute_forward_matrix,
    compute_color_matrix, 
    format_for_dng,
    compute_error_stats,
    compute_wb,
)



class ImageProcessor:
    """
    Main image processing API with calibration methods and parallel processing support.
    OPTIMIZED VERSION.
    """
    
    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize the image processor.
        
        Args:
            n_workers: Number of parallel workers (default: cpu_count() - 1)
        """
        self.color_matrix = None
        self.forward_matrix = None
        self.wb_gains = None
        self.reference_colors = COLORCHECKER_REFERENCE_SRGB.copy()
        
    
    def _load_and_demosaic(self, raw_path: Path) -> np.ndarray:
        """Load RAW image and demosaic."""
        print(f"\nLoading RAW image: {raw_path.name}")
        nparray = load_raw_image(raw_path)
        rgb = demosaic_image(nparray)
        print(f"  Size: {rgb.shape[1]}x{rgb.shape[0]} pixels")
        return rgb
    
    def _isolate_region(self, 
                       full_image: np.ndarray,
                       bounds: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
        """Isolate a region from the full image."""
        (x1, y1), (x2, y2) = bounds
        isolated = full_image[y1:y2, x1:x2].copy()
        print(f"\nIsolated region: {x2-x1}x{y2-y1} pixels")
        return isolated
    
    def _select_patches_interactive(self,
                                    isolated: np.ndarray,
                                    display_scale: Optional[float] = None
                                    ) -> Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """Run interactive patch selection GUI."""
        print("\nStarting interactive patch selection...")
        selector = MultiPatchSelector(isolated, display_scale, num_patches=24)
        patch_coords = selector.select_patches()
        
        if patch_coords is None:
            print("Patch selection cancelled")
            return None
        
        print(f"✓ Selected {len(patch_coords)} patches")
        return patch_coords
    
    def _extract_patch_colors(self,
                             isolated: np.ndarray,
                             patch_coords: List[Tuple[Tuple[int, int], Tuple[int, int]]]
                             ) -> np.ndarray:
        """Extract average color from each patch."""
        print("\nExtracting patch colors...")
        colors = np.empty((len(patch_coords), 3), dtype=np.float64)
        
        for i, ((x1, y1), (x2, y2)) in enumerate(patch_coords):
            patch = isolated[y1:y2, x1:x2]
            # Faster than np.mean with axis tuple
            colors[i] = patch.reshape(-1, 3).mean(axis=0)
            print(f"  Patch {i+1:2d}: R={colors[i,0]:.3f}, G={colors[i,1]:.3f}, B={colors[i,2]:.3f}")
        
        return colors
    
    def _check_clipping(self,
                       isolated: np.ndarray,
                       patch_coords: List[Tuple[Tuple[int, int], Tuple[int, int]]]
                       ) -> List[int]:
        """Check which patches are clipped/overexposed."""
        print("\nChecking for clipped patches...")
        clipped = []
        
        for i, ((x1, y1), (x2, y2)) in enumerate(patch_coords):
            patch = isolated[y1:y2, x1:x2]
            max_val = patch.max()
            clipped_pixels = (patch >= 0.99).sum()
            clipped_pct = (clipped_pixels / patch.size) * 100
            
            if clipped_pct > 1:
                print(f"  Patch {i+1:2d}: ⚠ CLIPPED ({clipped_pct:.1f}%)")
                clipped.append(i)
            else:
                print(f"  Patch {i+1:2d}: ✓ OK (max={max_val:.3f})")
        
        return clipped
    
    def _get_references(self, adjust_white: bool = False) -> np.ndarray:
        """Get reference colors, optionally adjusting white patch."""
        refs = self.reference_colors.copy()
        if adjust_white:
            refs[18] = [0.99, 0.99, 0.99]
        return refs
    
    def _compute_matrix(self,
                       measured: np.ndarray,
                       reference: np.ndarray,
                       exclude_patches: List[int] = None) -> np.ndarray:
        """Compute color correction matrix using least squares."""
        print("\nComputing color correction matrix...")
        
        
        if exclude_patches is None:
            exclude_patches = []
        
        # 1) Reference sRGB -> linear -> XYZ
        reference_lin = srgb_to_linear(reference)
        reference_xyz = srgb_to_xyz_d65(reference_lin)

        # 2) Solve for ForwardMatrix (camera -> XYZ)
        F = compute_forward_matrix(measured, reference_xyz)

        # 3) Compute ColorMatrix (XYZ -> camera)
        C = compute_color_matrix(F)

        WB = compute_wb(measured)

        # 4) Format for DNG tags
        forward_str = format_for_dng(F, transpose=False, decimals=6)
        color_str = format_for_dng(C, transpose=False, decimals=6)

        print("# ForwardMatrix1/2 (camera -> XYZ, D65)")
        print(forward_str)
        print()
        print("# ColorMatrix1/2 (XYZ -> camera)")
        print(color_str)

        # 5) Error stats
        compute_error_stats(measured, reference_xyz, F)

        return C, F, WB
    
    def _test_correction(self,
                        measured: np.ndarray,
                        matrix: np.ndarray) -> np.ndarray:
        """Apply color correction to measured colors."""
        corrected = measured @ matrix.T
        corrected = np.clip(corrected, 0, None)
        return corrected
    
    def _calculate_errors(self,
                         measured: np.ndarray,
                         corrected: np.ndarray,
                         reference: np.ndarray) -> Dict[str, float]:
        """Calculate color errors before and after correction."""
        print("\nCalculating errors...")
        
        # Delta E in RGB space (vectorized)
        delta_before = np.sqrt(np.sum((measured * 255 - reference * 255) ** 2, axis=1))
        delta_after = np.sqrt(np.sum((corrected * 255 - reference * 255) ** 2, axis=1))
        
        errors = {
            'mean_before': delta_before.mean(),
            'max_before': delta_before.max(),
            'mean_after': delta_after.mean(),
            'max_after': delta_after.max()
        }
        
        improvement = ((errors['mean_before'] - errors['mean_after']) / 
                      errors['mean_before'] * 100)
        
        print(f"  Before: Mean ΔE = {errors['mean_before']:.2f}, Max = {errors['max_before']:.2f}")
        print(f"  After:  Mean ΔE = {errors['mean_after']:.2f}, Max = {errors['max_after']:.2f}")
        print(f"  Improvement: {improvement:.1f}%")
        
        return errors
    
    def _save_visualization(self, result: CalibrationResult, isolated: np.ndarray, output_path: Path):
        """Save visualization of patches with boxes."""
        from svs_raw_api.selection import save_patch_visualization
        save_patch_visualization(
            isolated, 
            result.patch_coords,
            display_scale=0.3,
            output_path=str(output_path)
        )
    
    
    def _print_calibration_summary(self, result: CalibrationResult):
        """Print calibration summary."""
        print("\n" + "="*70)
        print("CALIBRATION SUMMARY")
        print("="*70)
        print(f"Timestamp: {result.timestamp}")
        print(f"Mean error before: {result.mean_error_before:.2f}")
        print(f"Mean error after: {result.mean_error_after:.2f}")
        print(f"Improvement: {(result.mean_error_before - result.mean_error_after) / result.mean_error_before * 100:.1f}%")
        if result.clipped_patches:
            print(f"Clipped patches: {[i+1 for i in result.clipped_patches]}")
        print("="*70)
    

    def calibrate(self,
                                   calib_config: CalibrationConfig) -> CalibrationResult:
        """
        Full calibration workflow from ColorChecker RAW image.
        
        This method orchestrates all the calibration steps:
        1. Load and demosaic RAW image
        2. Isolate ColorChecker region
        3. Interactive patch selection
        4. Extract patch colors
        5. Check for clipping
        6. Compute color correction matrix
        7. Optionally compute white balance
        8. Test correction and calculate errors
        9. Export results
        
        Args:
            calib_config: CalibrationConfig with all settings
            
        Returns:
            CalibrationResult with matrix, errors, and metadata
        """
        # Load and demosaic
        full_image = self._load_and_demosaic(calib_config.colorchecker_raw_path)
        
        # Isolate ColorChecker
        if calib_config.checker_top_left is None or calib_config.checker_bottom_right is None:
            raise ValueError("ColorChecker bounds must be specified")
        
        bounds = (calib_config.checker_top_left, calib_config.checker_bottom_right)
        isolated_checker = self._isolate_region(full_image, bounds)
        
        # Interactive patch selection
        patch_coords = self._select_patches_interactive(isolated_checker, calib_config.display_scale)
        if patch_coords is None:
            raise ValueError("Patch selection was cancelled")
        
        # Extract patch colors
        measured_colors = self._extract_patch_colors(isolated_checker, patch_coords)
        
        # Check for clipping
        clipped_patches = self._check_clipping(isolated_checker, patch_coords)
        
        # Get reference colors
        reference = self._get_references(adjust_white=calib_config.adjust_white)
        
        # Exclude patches as needed
        exclude = []
        if calib_config.exclude_white:
            exclude.append(18)
        exclude.extend(clipped_patches)
        
        # Compute color correction matrix
        color_matrix, forward_matrix, wb_gains = self._compute_matrix(measured_colors, reference, exclude_patches=exclude)
        
        # Test correction
        corrected_colors = self._test_correction(measured_colors, color_matrix)
        
        # Calculate errors
        errors = self._calculate_errors(measured_colors, corrected_colors, reference)
        
        # Create result
        result = CalibrationResult(
            color_matrix=color_matrix,
            forward_matrix=forward_matrix,
            wb_gains=wb_gains,
            measured_colors=measured_colors,
            corrected_colors=corrected_colors,
            reference_colors=reference,
            patch_coords=patch_coords,
            mean_error_before=errors['mean_before'],
            mean_error_after=errors['mean_after'],
            max_error_before=errors['max_before'],
            max_error_after=errors['max_after'],
            clipped_patches=clipped_patches,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            output_dir=calib_config.output_dir,
            output_path=calib_config.output_path
        )
        
        # Print summary
        self._print_calibration_summary(result)

        # Save comparison and visualization
        vis_output_path = calib_config.output_dir / "plots" / f"calibration_patches_{result.timestamp}.png"
        vis_output_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_visualization(result, isolated_checker, vis_output_path)
        
        return result

   