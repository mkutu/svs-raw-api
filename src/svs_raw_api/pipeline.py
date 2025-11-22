"""
Agricultural Image Processing Pipeline API (OPTIMIZED)
For SVS-Vistek shr661CXGE camera with inspec.x L 4/60 lens

This module provides a clean API for RAW image processing with ColorChecker calibration.
OPTIMIZED VERSION with parallel processing support.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from datetime import datetime
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
import warnings

from svs_raw_api.constants import COLORCHECKER_REFERENCE_SRGB

from svs_raw_api.data import (
    ProcessingConfig,
    CalibrationConfig,
    CalibrationResult,
    ProcessingResult,
    BatchResult
)

from svs_raw_api.selection import MultiPatchSelector
from svs_raw_api.processing_utils import (
    load_raw_image, demosaic_image, apply_color_correction,
    apply_exposure_compensation, apply_tone_curve, 
    apply_highlight_rolloff
)


def _process_single_image_worker(args: Tuple) -> ProcessingResult:
    """
    Worker function for parallel processing.
    Must be at module level for pickling.
    """
    input_path, params, quiet = args
    
    try:
        # Load and demosaic
        nparray = load_raw_image(input_path)
        rgb = demosaic_image(nparray)
        
        # Color correction
        if params.color_matrix is not None:
            rgb = apply_color_correction(rgb, params.color_matrix)
        
        # Exposure
        if params.exposure_stops != 0.0:
            rgb = apply_exposure_compensation(rgb, params.exposure_stops)
        
        # Tone mapping
        if params.tone_mapping == 'rolloff':
            rgb = apply_highlight_rolloff(rgb, params.highlight_threshold, params.highlight_smoothness)
        elif params.tone_mapping == 'reinhard':
            rgb = apply_tone_curve(rgb, 'reinhard')
        elif params.tone_mapping == 'aces':
            rgb = apply_tone_curve(rgb, 'aces')
        elif params.tone_mapping == 'reinhard_extended':
            rgb = apply_tone_curve(rgb, 'reinhard_extended', white_point=params.white_point)
        
        # Gamma
        if params.gamma != 1.0:
            rgb = rgb ** params.gamma
        
        # Clip final output
        rgb = np.clip(rgb, 0, 1)
        
        # Save output
        output_dir = params.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Full resolution output
        fullres_path = None
        if params.output_fullres:
            fullres_path = output_dir / f"{input_path.stem}_full.{params.output_format}"
            _save_image(rgb, fullres_path, params.output_format, params.output_quality, params.output_bit_depth)
        
        # Preview output
        preview_path = None
        if params.output_preview:
            preview_h = int(rgb.shape[0] * params.preview_scale)
            preview_w = int(rgb.shape[1] * params.preview_scale)
            rgb_preview = cv2.resize(rgb, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
            preview_path = output_dir / f"{input_path.stem}_preview.{params.output_format}"
            _save_image(rgb_preview, preview_path, params.output_format, params.output_quality, params.output_bit_depth)
        
        output_path = fullres_path if fullres_path else preview_path
        
        return ProcessingResult(
            input_path=str(input_path),
            output_path=str(output_path),
            preview_path=str(preview_path) if preview_path else None,
            status='success'
        )
        
    except Exception as e:
        if not quiet:
            warnings.warn(f"Error processing {input_path.name}: {e}")
        return ProcessingResult(
            input_path=str(input_path),
            output_path='',
            preview_path=None,
            status='error',
            error=str(e)
        )


def _save_image(rgb: np.ndarray, path: Path, format: str, quality: int, bit_depth: int):
    """Save image to disk."""
    if bit_depth == 8:
        img_out = (rgb * 255).astype(np.uint8)
    else:
        img_out = (rgb * 65535).astype(np.uint16)
    
    img_bgr = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
    
    if format == 'jpg':
        cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(path), img_bgr)


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
        self.wb_gains = None
        self.reference_colors = COLORCHECKER_REFERENCE_SRGB.copy()
        
        # Set up parallel processing
        if n_workers is None:
            self.n_workers = max(1, cpu_count() - 1)
        else:
            self.n_workers = max(1, n_workers)
        
        print(f"ImageProcessor initialized with {self.n_workers} workers")
    
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
        
        # Create mask for patches to use
        mask = np.ones(len(measured), dtype=bool)
        mask[exclude_patches] = False
        
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
            avg = patch.reshape(-1, 3).mean(axis=0)
            patch_values.append(avg)
        
        patch_values = np.array(patch_values)
        
        # Average across neutral patches
        avg_neutral = patch_values.mean(axis=0)
        r, g, b = avg_neutral
        
        # Calculate gains (normalize to green)
        gains = {
            'r_gain': g / r if r > 0 else 1.0,
            'g_gain': 1.0,
            'b_gain': g / b if b > 0 else 1.0
        }
        
        print(f"  R gain: {gains['r_gain']:.3f}")
        print(f"  G gain: {gains['g_gain']:.3f}")
        print(f"  B gain: {gains['b_gain']:.3f}")
        
        return gains
    
    def _save_visualization(self, result: CalibrationResult, isolated: np.ndarray, output_path: Path):
        """Save visualization of patches with boxes."""
        from image_processing_api.selection import save_patch_visualization
        save_patch_visualization(
            isolated, 
            result.patch_coords,
            display_scale=0.3,
            output_path=str(output_path)
        )
    
    def _save_comparison(self, result: CalibrationResult, isolated: np.ndarray, output_path: Path):
        """Save before/after comparison."""
        from image_processing_api.selection import save_comparison_image
        
        # Apply correction to isolated region for comparison
        corrected = apply_color_correction(isolated, result.color_matrix)
        corrected = np.clip(corrected, 0, 1)
        
        save_comparison_image(
            isolated,
            corrected,
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
    
    def _export_image_output(self, output_path: Path, params: ProcessingConfig, rgb: np.ndarray) -> Optional[Path]:
        """Export image with full resolution and/or preview."""
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        preview_path = None
        
        # Full resolution
        if params.output_fullres:
            _save_image(rgb, output_path, params.output_format, params.output_quality, params.output_bit_depth)
            print(f"  ✓ Saved full: {output_path.name}")
        
        # Preview
        if params.output_preview:
            preview_h = int(rgb.shape[0] * params.preview_scale)
            preview_w = int(rgb.shape[1] * params.preview_scale)
            rgb_preview = cv2.resize(rgb, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
            
            preview_path = output_path.parent / f"{output_path.stem}_preview{output_path.suffix}"
            _save_image(rgb_preview, preview_path, params.output_format, params.output_quality, params.output_bit_depth)
            print(f"  ✓ Saved preview: {preview_path.name}")
        
        return preview_path
    
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
        color_matrix = self._compute_matrix(measured_colors, reference, exclude_patches=exclude)
        
        # Compute white balance if requested
        wb_gains = {}
        if calib_config.calc_wb:
            wb_gains = self._compute_white_balance(isolated_checker, patch_coords)
        
        # Test correction
        corrected_colors = self._test_correction(measured_colors, color_matrix)
        
        # Calculate errors
        errors = self._calculate_errors(measured_colors, corrected_colors, reference)
        
        # Create result
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
            params: ProcessingConfig configuration
            
        Returns:
            ProcessingResult with status
        """
        return _process_single_image_worker((input_path, params, False))
    
    def process_batch(self,
                     input_dir: Path,
                     params: ProcessingConfig,
                     pattern: str = "*.RAW",
                     skip_first: int = 0,
                     limit: Optional[int] = None,
                     use_parallel: bool = True,
                     show_progress: bool = True) -> BatchResult:
        """
        Process batch of RAW images with parallel processing.
        
        Args:
            input_dir: Input directory
            params: ProcessingConfig configuration
            pattern: File pattern (default "*.RAW")
            skip_first: Skip first N files
            limit: Max files to process
            use_parallel: Use parallel processing (default True)
            show_progress: Show progress bar (default True)
            
        Returns:
            BatchResult with statistics
        """
        if params.color_matrix is None:
            raise ValueError("No calibration loaded - set params.color_matrix")
        
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
        print(f"Workers: {self.n_workers if use_parallel else 1}")
        print(f"Gamma: {params.gamma}, Exposure: {params.exposure_stops:+.2f}, Tone: {params.tone_mapping}")
        print("="*70)
        
        if len(raw_files) == 0:
            return BatchResult(0, 0, 0, [], str(params.output_dir))
        
        # Prepare arguments for workers
        task_args = [(f, params, True) for f in raw_files]
        
        # Process files
        results = []
        
        if use_parallel and self.n_workers > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                if show_progress:
                    # With progress bar
                    futures = {executor.submit(_process_single_image_worker, args): args[0] 
                             for args in task_args}
                    
                    with tqdm(total=len(raw_files), desc="Processing", unit="image") as pbar:
                        for future in as_completed(futures):
                            result = future.result()
                            results.append(result)
                            
                            if result.status == 'success':
                                pbar.set_postfix_str(f"✓ {Path(result.input_path).name}")
                            else:
                                pbar.set_postfix_str(f"✗ {Path(result.input_path).name}")
                            pbar.update(1)
                else:
                    # Without progress bar
                    futures = [executor.submit(_process_single_image_worker, args) 
                             for args in task_args]
                    for future in as_completed(futures):
                        results.append(future.result())
        else:
            # Sequential processing
            if show_progress:
                task_iter = tqdm(task_args, desc="Processing", unit="image")
            else:
                task_iter = task_args
            
            for args in task_iter:
                result = _process_single_image_worker(args)
                results.append(result)
                
                if show_progress:
                    if result.status == 'success':
                        task_iter.set_postfix_str(f"✓ {Path(result.input_path).name}")
                    else:
                        task_iter.set_postfix_str(f"✗ {Path(result.input_path).name}")
        
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
                       output_root: Path,
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
            combo_str = "_".join(f"{n}={v}" for n, v in zip(param_names, combo))
            config.output_dir = output_root / combo_str
            
            print(f"\n[{i}/{len(combinations)}] {combo_str}")
            
            # Process batch
            result = self.process_batch(
                input_dir=input_dir,
                params=config,
                pattern=pattern,
                limit=limit,
                use_parallel=True,
                show_progress=False
            )
            
            sweep_results.append({
                'parameters': dict(zip(param_names, combo)),
                'result': result
            })
        
        # Save sweep summary
        summary_path = output_root / "sweep_summary.json"
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