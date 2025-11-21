"""
Agricultural Image Processing Pipeline API
For SVS-Vistek shr661CXGE camera with inspec.x L 4/60 lens

This module provides a clean API for RAW image processing with ColorChecker calibration.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from datetime import datetime
import json

from image_processing_api.constants import COLORCHECKER_REFERENCE_SRGB

from image_processing_api.data import (
    ProcessingConfig,
    CalibrationConfig,
    CalibrationResult,
    ProcessingResult,
    BatchResult
)

from image_processing_api.selection import MultiPatchSelector  # Import your existing class
from image_processing_api.archive.image_processing import (
    load_raw_image, demosaic_image, apply_color_correction, 
    apply_exposure_compensation, apply_highlight_rolloff, apply_tone_curve, 
    check_clipping_stats
)


class ImageProcessor:
    """
    Main image processing API with calibration methods broken into reusable pieces.
    """
    
    def __init__(self):
        self.color_matrix = None
        self.wb_gains = None
        self.reference_colors = COLORCHECKER_REFERENCE_SRGB.copy()
    
    def load_and_demosaic(self, raw_path: Path) -> np.ndarray:
        """Load RAW image and demosaic."""
        nparray = load_raw_image(raw_path)
        
        rgb = demosaic_image(nparray)
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
        """
        Run interactive patch selection GUI.
        
        This uses your existing MultiPatchSelector class.
        """
        
        
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
        colors = []
        
        for i, ((x1, y1), (x2, y2)) in enumerate(patch_coords):
            patch = isolated[y1:y2, x1:x2]
            avg_color = np.mean(patch, axis=(0, 1))
            colors.append(avg_color)
            print(f"  Patch {i+1:2d}: R={avg_color[0]:.3f}, G={avg_color[1]:.3f}, B={avg_color[2]:.3f}")
        
        return np.array(colors)
    
    def _check_clipping(self,
                       isolated: np.ndarray,
                       patch_coords: List[Tuple[Tuple[int, int], Tuple[int, int]]]
                       ) -> List[int]:
        """Check which patches are clipped/overexposed."""
        print("\nChecking for clipped patches...")
        clipped = []
        
        for i, ((x1, y1), (x2, y2)) in enumerate(patch_coords):
            patch = isolated[y1:y2, x1:x2]
            max_val = np.max(patch)
            clipped_pixels = np.sum(patch >= 0.99)
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
            refs[18] = [0.99, 0.99, 0.99]  # Adjust white patch
        return refs
    
    def _compute_matrix(self,
                       measured: np.ndarray,
                       reference: np.ndarray,
                       exclude_patches: List[int] = None) -> np.ndarray:
        """Compute color correction matrix using least squares."""
        print("\nComputing color correction matrix...")
        
        if exclude_patches is None:
            exclude_patches = []
        
        # Create mask for patches to use
        mask = np.ones(len(measured), dtype=bool)
        for idx in exclude_patches:
            mask[idx] = False
        
        # Use only non-excluded patches
        measured_subset = measured[mask]
        reference_subset = reference[mask]
        
        print(f"  Using {len(measured_subset)}/24 patches")
        if exclude_patches:
            print(f"  Excluded patches: {[i+1 for i in exclude_patches]}")
        
        # Compute matrix: measured @ M^T = reference
        M = np.linalg.lstsq(measured_subset, reference_subset, rcond=None)[0].T
        
        print("\nColor correction matrix:")
        print(M)
        
        return M
    
    def _test_correction(self,
                        measured: np.ndarray,
                        matrix: np.ndarray) -> np.ndarray:
        """Apply color correction to measured colors."""
        corrected = measured @ matrix.T
        corrected = np.clip(corrected, 0, None)  # Clip negatives only
        return corrected
    
    def _calculate_errors(self,
                         measured: np.ndarray,
                         corrected: np.ndarray,
                         reference: np.ndarray) -> Dict[str, float]:
        """Calculate color errors before and after correction."""
        print("\nCalculating errors...")
        
        # Delta E in RGB space
        delta_before = np.sqrt(np.sum((measured * 255 - reference * 255) ** 2, axis=1))
        delta_after = np.sqrt(np.sum((corrected * 255 - reference * 255) ** 2, axis=1))
        
        errors = {
            'mean_before': np.mean(delta_before),
            'max_before': np.max(delta_before),
            'mean_after': np.mean(delta_after),
            'max_after': np.max(delta_after)
        }
        
        improvement = ((errors['mean_before'] - errors['mean_after']) / 
                      errors['mean_before'] * 100)
        
        print(f"  Before: Mean ΔE = {errors['mean_before']:.2f}, Max = {errors['max_before']:.2f}")
        print(f"  After:  Mean ΔE = {errors['mean_after']:.2f}, Max = {errors['max_after']:.2f}")
        print(f"  Improvement: {improvement:.1f}%")
        
        return errors

    def _compute_white_balance(self,
                              isolated: np.ndarray,
                              patch_coords: List[Tuple[Tuple[int, int], Tuple[int, int]]]
                              ) -> Dict[str, float]:
        """Compute white balance gains from neutral patches."""
        print("\nComputing white balance from neutral patches...")
        
        # Neutral patches are indices 18-23 (White, Neutral 8, 6.5, 5, 3.5, Black)
        neutral_indices = [18, 19, 20, 21, 22, 23]
        
        patch_values = []
        for i in neutral_indices:
            (x1, y1), (x2, y2) = patch_coords[i]
            patch = isolated[y1:y2, x1:x2]
            r_mean = np.mean(patch[:, :, 0])
            g_mean = np.mean(patch[:, :, 1])
            b_mean = np.mean(patch[:, :, 2])
            patch_values.append([r_mean, g_mean, b_mean])
        
        # Average across neutral patches
        patch_values = np.array(patch_values)
        avg_r = np.mean(patch_values[:, 0])
        avg_g = np.mean(patch_values[:, 1])
        avg_b = np.mean(patch_values[:, 2])
        
        # Normalize to green (most accurate due to 2x sampling in Bayer)
        r_gain = avg_g / avg_r
        g_gain = 1.0
        b_gain = avg_g / avg_b
        
        wb_gains = {
            'R_gain': float(r_gain),
            'G_gain': float(g_gain),
            'B_gain': float(b_gain)
        }
        
        print(f"  R gain: {r_gain:.4f}")
        print(f"  G gain: {g_gain:.4f}")
        print(f"  B gain: {b_gain:.4f}")
        
        # Sanity check
        if any(g < 0.3 or g > 3.0 for g in [r_gain, g_gain, b_gain]):
            print("  ⚠ WARNING: Unusual white balance gains!")
        else:
            print("  ✓ White balance gains look reasonable")
        
        return wb_gains

    def _print_calibration_summary(self, result: CalibrationResult):
        """Print summary of calibration results."""
        print("\n" + "="*70)
        print("CALIBRATION COMPLETE")
        print("="*70)
        print(f"Mean ΔE improvement: {result.mean_error_before:.2f} → {result.mean_error_after:.2f}")
        if result.wb_gains:
            print(f"White balance: R={result.wb_gains['R_gain']:.4f}, "
              f"G={result.wb_gains['G_gain']:.4f}, B={result.wb_gains['B_gain']:.4f}")
        if result.clipped_patches:
            print(f"Clipped patches: {[i+1 for i in result.clipped_patches]}")
        print("="*70)
    
    def load_calibration(self, matrix_path: Path, json_path: Optional[Path] = None):
        """Load pre-computed calibration."""
        self.color_matrix = np.load(matrix_path)
        print(f"Loaded color matrix from: {matrix_path}")
        
        if json_path and json_path.exists():
            with open(json_path, 'r') as f:
                metadata = json.load(f)
                self.wb_gains = metadata.get('wb_gains', {})
                print(f"Loaded white balance: R={self.wb_gains.get('R_gain', 1.0):.4f}, "
                      f"G={self.wb_gains.get('G_gain', 1.0):.4f}, "
                      f"B={self.wb_gains.get('B_gain', 1.0):.4f}")
                
    def _save_patch_visualization(
            self,
            isolated_checker: np.ndarray, 
            patches: List[Tuple[Tuple[int, int], Tuple[int, int]]],
            display_scale: float = 1.0,
            output_path: str = 'selected_patches.png'):
        
        """Save visualization of selected patches on ColorChecker."""
        h, w = isolated_checker.shape[:2]
        vis_image = cv2.resize(isolated_checker, (int(w * display_scale), int(h * display_scale)))
        vis_image = (vis_image * 255).astype(np.uint8).copy()
        
        for i, (top_left, bottom_right) in enumerate(patches):
            x1_s = int(top_left[0] * display_scale)
            y1_s = int(top_left[1] * display_scale)
            x2_s = int(bottom_right[0] * display_scale)
            y2_s = int(bottom_right[1] * display_scale)
            
            cv2.rectangle(vis_image, (x1_s, y1_s), (x2_s, y2_s), (0, 255, 0), 2)
            cv2.putText(vis_image, str(i + 1), 
                    (x1_s + 5, y1_s + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_bgr)
        print(f"✓ Saved visualization: {output_path}")
                
    def _save_visualization(self,
                          result: CalibrationResult,
                          isolated: np.ndarray,
                          output_path: Path):
        """Save visualization of selected patches."""
        # Use your existing save_patch_visualization function
        self._save_patch_visualization(
            isolated, 
            result.patch_coords, 
            display_scale=0.5, 
            output_path=str(output_path)
        )
    
    def _save_comparison_image(
            self,
            original: np.ndarray, 
            corrected: np.ndarray, 
            display_scale: float = 1.0,
            output_path: str = 'colorchecker_before_after.png'
            ):
    
        """Save side-by-side comparison."""
        orig_scaled = cv2.resize(original, (0, 0), fx=display_scale, fy=display_scale)
        corr_scaled = cv2.resize(corrected, (0, 0), fx=display_scale, fy=display_scale)
        
        orig_scaled = (orig_scaled * 255).astype(np.uint8)
        corr_scaled = (corr_scaled * 255).astype(np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(orig_scaled, 'Original', (20, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(corr_scaled, 'Corrected', (20, 40), font, 1, (255, 255, 255), 2)
        
        comparison = np.hstack([orig_scaled, corr_scaled])
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(output_path, comparison_bgr)
        print(f"✓ Saved comparison: {output_path}")

    def _save_comparison(self,
                       result: CalibrationResult,
                       isolated: np.ndarray,
                       output_path: Path):
        """Save before/after comparison."""
        # Apply correction to isolated region
        corrected = isolated @ result.color_matrix.T
        corrected = np.clip(corrected, 0, 1)
        
        # Use your existing save_comparison_image function
        self._save_comparison_image(isolated, corrected, display_scale=0.5, output_path=str(output_path))

    def _export_image_output(self, output_path: Path, params: ProcessingConfig, rgb: np.ndarray) -> Optional[Path]:
        """Export processed image according to parameters."""
        print(f"\nExporting image to: {output_path}")
        rgb_8bit = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2BGR)
        
        if params.output_fullres:
            if params.output_format == 'jpg':
                cv2.imwrite(str(output_path), bgr, 
                            [cv2.IMWRITE_JPEG_QUALITY, params.output_quality], 
                            [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444],
                            [cv2.IMWRITE_JPEG_CHROMA_QUALITY , 100]
                )
            else:
                print(f"  ⚠ Unsupported output format: {params.output_format}")
                
            
            # Preview
        preview_path = None
        if params.output_preview:
            preview_path = output_path.parent / f"{output_path.stem}_prev{output_path.suffix}"
            resized = cv2.resize(bgr, (0, 0), fx=params.preview_scale, fy=params.preview_scale)
            cv2.imwrite(str(preview_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  ✓ Saved preview: {preview_path.name}")
        return preview_path

    def calibrate(self,
                  calib_config: CalibrationConfig) -> CalibrationResult:
        """
        Create color correction matrix from ColorChecker.
        
        This is the main entry point that orchestrates all the calibration steps.
        
        Args:
            calib_config: CalibrationConfig with all parameters
            
        Returns:
            CalibrationResult with matrix, colors, and metrics
        """
        print("\n" + "="*70)
        print("COLOR CALIBRATION")
        print("="*70)
        
        
        # Step 1: Load and process the ColorChecker image
        full_image = self.load_and_demosaic(calib_config.colorchecker_raw_path)
        
        # Step 2: Isolate ColorChecker region
        checker_bounds = (calib_config.checker_top_left, calib_config.checker_bottom_right)
        isolated_checker = self._isolate_region(full_image, checker_bounds)
        
        # Step 3: Interactive patch selection
        patch_coords = self._select_patches_interactive(
            isolated_checker, 
            calib_config.display_scale
        )
        
        if patch_coords is None:
            raise ValueError("Calibration cancelled by user")
        
        # Step 4: Extract colors from patches
        measured_colors = self._extract_patch_colors(
            isolated_checker, 
            patch_coords
        )
        
        # Step 5: Check for clipping
        clipped_patches = self._check_clipping(isolated_checker, patch_coords)
        
        # Step 6: Get reference colors
        reference = self._get_references(calib_config.adjust_white)
        
        # Step 7: Compute color correction matrix
        color_matrix = self._compute_matrix(
            measured_colors,
            reference,
            exclude_patches=[18] if calib_config.exclude_white else []
        )
        
        # Step 8: Test correction
        corrected_colors = self._test_correction(
            measured_colors, 
            color_matrix
        )
        
        # Step 9: Calculate errors
        errors = self._calculate_errors(
            measured_colors,
            corrected_colors,
            reference
        )
        if calib_config.calc_wb:
            # Step 10: Compute white balance from neutral patches
            wb_gains = self._compute_white_balance(
                isolated_checker,
                patch_coords
            )
        else:
            wb_gains = None
        
        # Step 11: Store results
        self.color_matrix = color_matrix
        self.wb_gains = wb_gains
        
        # Step 12: Create result object
        result = CalibrationResult(
            color_matrix=color_matrix,
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
        )
        
        # Print summary
        self._print_calibration_summary(result)

        # Save comparison and visualization
        vis_output_path = calib_config.output_dir / "plots" / f"calibration_patches_{result.timestamp}.png"
        vis_output_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_visualization(result, isolated_checker, vis_output_path)
        comp_output_path = calib_config.output_dir / "plots" / f"calibration_comparison_{result.timestamp}.png"
        comp_output_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_comparison(result, isolated_checker, comp_output_path)
        
        return result

    def process_image(self,
                     input_path: Path,
                     params: ProcessingConfig) -> ProcessingResult:
        """
        Process a single RAW image.
        
        Args:
            input_path: Path to RAW file
            output_path: Path for output
            params: ProcessingParams configuration
            
        Returns:
            ProcessingResult with status
        """
        color_matrix = params.color_matrix
        
        if color_matrix is None:
            raise ValueError("No calibration loaded")
        
        print(f"\nProcessing: {input_path.name}")
        
        try:
            # Load and demosaic
            rgb = self.load_and_demosaic(input_path)
            check_clipping_stats(rgb, "After demosaic")
            
            # Color correction
            rgb = apply_color_correction(rgb, color_matrix)
            check_clipping_stats(rgb, "After color correction")
            
            # Exposure
            if params.exposure_stops != 0.0:
                rgb = apply_exposure_compensation(rgb, params.exposure_stops)
                check_clipping_stats(rgb, f"After exposure {params.exposure_stops:+.2f}")
            
            # Tone mapping
            if params.tone_mapping == 'rolloff':
                rgb = apply_highlight_rolloff(rgb, params.highlight_threshold, params.highlight_smoothness)
                check_clipping_stats(rgb, "After rolloff")
            
            elif params.tone_mapping == 'reinhard':
                rgb = apply_tone_curve(rgb, 'reinhard')
                check_clipping_stats(rgb, "After reinhard")
            
            elif params.tone_mapping == 'aces':
                rgb = apply_tone_curve(rgb, 'aces')
                check_clipping_stats(rgb, "After ACES")
            
            elif params.tone_mapping == 'reinhard_extended':
                rgb = apply_tone_curve(rgb, 'reinhard_extended', white_point=params.white_point)
                check_clipping_stats(rgb, "After reinhard extended")
            
            elif params.tone_mapping != 'none':
                raise ValueError(f"Unknown tone mapping: {params.tone_mapping}")
            
            # Gamma
            if params.gamma != 1.0:
                rgb = rgb ** params.gamma
                check_clipping_stats(rgb, f"After gamma {params.gamma}")
            
            # Save
            output_dir = params.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            if params.output_fname_prefix:
                output_path = output_dir / f"{params.output_fname_prefix}_{input_path.stem}.{params.output_format}"
            else:
                output_path = output_dir / f"{input_path.stem}.{params.output_format}"
            preview_path = self._export_image_output(output_path, params, rgb)
            
            return ProcessingResult(
                input_path=str(input_path),
                output_path=str(output_path),
                preview_path=str(preview_path) if preview_path else None,
                status='success'
            )
            
        except Exception as e:
            print(f"Error processing {input_path.name}: {e}")
            return ProcessingResult(
                input_path=str(input_path),
                output_path='',
                preview_path=None,
                status='error',
                error=str(e)
            )
    
    def process_batch(self,
                     input_dir: Path,
                     params: ProcessingConfig,
                     pattern: str = "*.RAW",
                     skip_first: int = 0,
                     limit: Optional[int] = None) -> BatchResult:
        """
        Process batch of RAW images.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            params: ProcessingParams configuration
            pattern: File pattern (default "*.RAW")
            skip_first: Skip first N files
            limit: Max files to process
            
        Returns:
            BatchResult with statistics
        """
        if self.color_matrix is None:
            raise ValueError("No calibration loaded")
        
        # Find files
        if not input_dir.exists():
            raise ValueError(f"Input directory not found: {input_dir}")
        
        raw_files = sorted(list(input_dir.glob(pattern)))
        if skip_first > 0:
            raw_files = raw_files[skip_first:]
        if limit is not None:
            raw_files = raw_files[:limit]
        
        print("\n" + "="*70)
        print("BATCH PROCESSING")
        print("="*70)
        print(f"Files: {len(raw_files)}")
        print(f"Gamma: {params.gamma}, Exposure: {params.exposure_stops:+.2f}, Tone: {params.tone_mapping}")
        print("="*70)
        
        if len(raw_files) == 0:
            return BatchResult(0, 0, 0, [], str(params.output_dir))
        
        # Process each file
        results = []
        for i, raw_file in enumerate(raw_files, 1):
            print(f"\n[{i}/{len(raw_files)}]", end=" ")
            
            ext = '.jpg' if params.output_format == 'jpg' else '.png'
            output_path = params.output_dir / f"{raw_file.stem}{ext}"
            
            result = self.process_image(raw_file, params)
            results.append(result)
            
            if result.status == 'success':
                print(f"  ✓ Saved: {output_path.name}")
            else:
                print(f"  ✗ Failed: {raw_file.name} - {result.error}")
        
        # Create result
        successful = sum(1 for r in results if r.status == 'success')
        failed = sum(1 for r in results if r.status == 'error')
        
        batch_result = BatchResult(
            total=len(raw_files),
            successful=successful,
            failed=failed,
            results=results,
            output_dir=str(params.output_dir.absolute())
        )
        
        batch_result.print_summary()
        return batch_result
    
    def parameter_sweep(self,
                       input_dir: Path,
                       sweep_config: Dict[str, List],
                       base_config: ProcessingConfig,
                       pattern: str = "*.RAW",
                       limit: Optional[int] = None) -> Dict[str, any]:
        """
        Run systematic parameter sweep across multiple values with parallel processing.
        
        Args:
            input_dir: Directory containing RAW files
            output_root: Root directory for organized outputs
            sweep_config: Dictionary defining parameter ranges, e.g.:
                {
                    'tone_mapping': ['rolloff', 'reinhard', 'aces'],
                    'gamma': [0.7, 0.85, 0.9],
                    'exposure_stops': [0.3, 0.5, 0.7]
                }
            base_config: Base ProcessingConfig to modify
            pattern: File pattern for RAW files
            limit: Limit number of files to process
                
        Returns:
            Dictionary with sweep results and output locations
        """
        import itertools
        
        # Generate all parameter combinations
        param_names = list(sweep_config.keys())
        param_values = list(sweep_config.values())
        combinations = list(itertools.product(*param_values))
        
        print(f"\nParameter Sweep: {len(combinations)} combinations")
        print(f"Parameters: {param_names}")
        
        sweep_results = []
        
        for i, combo in enumerate(combinations, 1):
            # Create config for this combination
            config = ProcessingConfig(
                **{k: v for k, v in base_config.__dict__.items() 
                   if k not in param_names}
            )
            
            # Apply sweep parameters
            for name, value in zip(param_names, combo):
                setattr(config, name, value)
            
            # Create output directory
            
            if combo[1] == 'rolloff':
                combo_str = f"roll_g{combo[0]}_ex{combo[2]}"
            elif combo[1] == 'reinhard':
                combo_str = f"rein_g{combo[0]}_ex{combo[2]}"
            elif combo[1] == 'aces':
                combo_str = f"aces_g{combo[0]}_ex{combo[2]}"
            else:
                combo_str = f"{combo[1]}_g{combo[0]}_ex{combo[2]}"
            
            if config.output_use_param_prefix:
                config.output_fname_prefix = combo_str
            else:
                config.output_dir = config.output_dir / combo_str

            
            # Process batch
            result = self.process_batch(
                input_dir=input_dir,
                params=config,
                pattern=pattern,
                limit=limit
            )
            
            sweep_results.append({
                'parameters': dict(zip(param_names, combo)),
                'result': result
            })
        
        # Save sweep summary
        summary_path = config.output_dir / "sweep_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'total_combinations': len(combinations),
                'parameters': param_names,
                'results': [
                    {
                        'params': r['parameters'],
                        'successful': r['result'].successful,
                        'failed': r['result'].failed,
                        'output_dir': r['result'].output_dir
                    }
                    for r in sweep_results
                ]
            }, f, indent=2)
        
        print(f"\n✓ Sweep complete! Summary saved to: {summary_path}")
        
        return {
            'combinations': len(combinations),
            'results': sweep_results,
            'summary_path': str(summary_path)
        }