"""
Comprehensive Parameter Sweep for RawTherapee Adjustments

This script systematically tests different combinations of all processing parameters
to help you find the optimal settings for your images.

You can choose to sweep:
1. Individual parameters (test one at a time)
2. Multiple parameters (test combinations)
3. Full factorial sweep (test all combinations - can be huge!)
"""

import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from itertools import product
from typing import Dict, List, Any, Tuple
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Import processing functions
from image_processing_api.archive.image_process_enhanced import apply_black_level, process_image_complete, print_stats
from image_processing_api.archive.image_processing import (
    apply_color_correction,
    load_raw_image,
    demosaic_image
)

from image_processing_api.rawtherapee_params import PARAMETER_GRIDS, get_baseline_params



def sweep_single_parameter(
    param_name: str,
    color_matrix_path: Path,
    input_dir: Path,
    output_base_dir: Path,
    n_images: int = 3
):
    """
    Sweep a single parameter while keeping others at baseline.
    
    This is useful for understanding the effect of one parameter at a time.
    """
    
    if param_name not in PARAMETER_GRIDS:
        raise ValueError(f"Unknown parameter: {param_name}")
    
    print("\n" + "="*70)
    print(f"SINGLE PARAMETER SWEEP: {param_name}")
    print("="*70)
    
    param_info = PARAMETER_GRIDS[param_name]
    values = param_info['values']
    
    print(f"Testing {len(values)} values: {values}")
    print(f"{param_info['description']}")
    print(f"Processing {n_images} images per value")
    print("="*70)
    
    # Load color matrix
    color_matrix = np.load(str(color_matrix_path))
    
    # Get baseline parameters
    baseline = get_baseline_params()
    
    # Get RAW files
    raw_files = sorted(list(input_dir.glob('*.RAW')))
    if len(raw_files) == 0:
        print("ERROR: No RAW files found!")
        return
    
    # Use subset
    raw_files = raw_files[1:n_images+1]  # Skip first, use next n
    
    print(f"\nProcessing {len(raw_files)} images:")
    for f in raw_files:
        print(f"  - {f.name}")
    
    # Sweep through parameter values
    results = []
    
    for i, value in enumerate(values, 1):
        print(f"\n[{i}/{len(values)}] Testing {param_name} = {value}")
        print("-" * 70)
        
        # Update parameter
        params = baseline.copy()
        params[param_name] = value
        
        # Create output directory
        output_dir = output_base_dir / param_name / f"{param_name}_{value}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        for raw_file in raw_files:
            print(f"  Processing: {raw_file.name}")
            
            # 0. Load RAW image
            nparray = load_raw_image(raw_file)
            print_stats(nparray, "0. After Loaded RAW array")
            
            # 1. Black level adjustment (linear light, before color correction)
            image = nparray.copy()
            if params['black_level'] != 0:
                image = apply_black_level(image, params['black_level'])
                print_stats(image, "1. After black level")
            
            # 2. Demosaicing
            demosaiced = demosaic_image(image)
            print_stats(demosaiced, "2. After Demosaicing")

            # 3. Color correction (linear light)
            image = apply_color_correction(image, color_matrix)
            print_stats(image, "3. After color correction")
            
            # Process
            processed = process_image_complete(
                image=image,
                exposure_stops=params['exposure_stops'],
                highlight_protection=params['highlight_protection'],
                highlight_threshold=params['highlight_threshold'],
                highlight_smoothness=params['highlight_smoothness'],
                highlight_strength=params['highlight_strength'],
                shadow_compression=params['shadow_compression'],
                shadow_threshold=params['shadow_threshold'],
                brightness=params['brightness'],
                contrast=params['contrast'],
                saturation=params['saturation'],
                gamma=params['gamma'],
                vignette_strength=params['vignette_strength'],
                vignette_feather=params['vignette_feather'],
                vignette_roundness=params['vignette_roundness'],
                verbose=False
            )
            
            # Save
            processed_8bit = (processed * 255).astype(np.uint8)
            processed_bgr = cv2.cvtColor(processed_8bit, cv2.COLOR_RGB2BGR)
            
            output_path = output_dir / f"{raw_file.stem}.jpg"
            cv2.imwrite(str(output_path), processed_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        results.append({
            'param_name': param_name,
            'param_value': value,
            'output_dir': str(output_dir)
        })
        
        print(f"  ✓ Saved to: {output_dir}")
    
    # Save sweep metadata
    metadata = {
        'sweep_type': 'single_parameter',
        'parameter': param_name,
        'values_tested': values,
        'baseline_params': baseline,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_path = output_base_dir / param_name / 'sweep_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✓ SWEEP COMPLETE: {param_name}")
    print("="*70)
    print(f"Results in: {output_base_dir / param_name}")
    print(f"Compare images to see effect of {param_name}")
    print("="*70)


def sweep_multiple_parameters(
    param_names: List[str],
    color_matrix_path: Path,
    input_dir: Path,
    output_base_dir: Path,
    n_images: int = 2, 
    preview_only: bool = True
):
    """
    Sweep multiple parameters simultaneously.
    
    Tests all combinations of the specified parameters.
    WARNING: Can create many combinations! 4 values × 3 params = 64 combinations
    """
    
    print("\n" + "="*70)
    print(f"MULTI-PARAMETER SWEEP")
    print("="*70)
    
    # Validate parameters
    for param in param_names:
        if param not in PARAMETER_GRIDS:
            raise ValueError(f"Unknown parameter: {param}")
    
    # Get parameter grids
    param_grids = {name: PARAMETER_GRIDS[name]['values'] for name in param_names}
    
    # Calculate total combinations
    total_combinations = 1
    for values in param_grids.values():
        total_combinations *= len(values)
    
    print(f"Parameters to sweep:")
    for name, values in param_grids.items():
        print(f"  - {name}: {values} ({len(values)} values)")
    
    print(f"\nTotal combinations: {total_combinations}")
    print(f"Processing {n_images} images per combination")
    print(f"Total images to process: {total_combinations * n_images}")
    
    if total_combinations > 100:
        print("\n⚠️  WARNING: This will create many combinations!")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Sweep cancelled.")
            return
    
    print("="*70)
    
    
    # Load color matrix
    color_matrix = np.load(str(color_matrix_path))
    
    # Get baseline parameters
    baseline = get_baseline_params()
    
    # Get RAW files
    raw_files = sorted(list(input_dir.glob('*.RAW')))
    # raw_files = raw_files[1:n_images+1]
    
    # Generate all combinations
    param_values = [param_grids[name] for name in param_names]
    combinations = list(product(*param_values))
    
    print(f"\nProcessing {len(combinations)} combinations...")
    
    results = []
    # Process images
    for raw_file in raw_files:
        # 0. Load RAW image
        nparray = load_raw_image(raw_file)
        print_stats(nparray, "0. After Loaded RAW array")
        
        # 1. Black level adjustment (linear light, before color correction)
        image = nparray.copy()
        if baseline['black_level'] != 0:
            image = apply_black_level(image, baseline['black_level'])
            print_stats(image, "1. After black level")
        
        # 2. Demosaicing
        demosaiced = demosaic_image(image)
        print_stats(demosaiced, "2. After Demosaicing")

        # 3. Color correction (linear light)
        image = apply_color_correction(demosaiced, color_matrix)
        print_stats(image, "3. After color correction")

        for i, combo in enumerate(combinations, 1):
            # Create parameter dict
            params = baseline.copy()
            combo_dict = dict(zip(param_names, combo))
            params.update(combo_dict)
            
            # Create descriptive folder name
            folder_name = "_".join([f"{name}_{combo_dict[name]}" for name in param_names])
            output_dir = output_base_dir / 'multi_param' / folder_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n[{i}/{len(combinations)}] " + ", ".join([f"{k}={v}" for k, v in combo_dict.items()]))
            
            
                
            processed = process_image_complete(
                image=image,
                exposure_stops=params['exposure_stops'],
                highlight_protection=params['highlight_protection'],
                highlight_threshold=params['highlight_threshold'],
                highlight_smoothness=params['highlight_smoothness'],
                highlight_strength=params['highlight_strength'],
                shadow_compression=params['shadow_compression'],
                shadow_threshold=params['shadow_threshold'],
                brightness=params['brightness'],
                contrast=params['contrast'],
                saturation=params['saturation'],
                gamma=params['gamma'],
                vignette_strength=params['vignette_strength'],
                vignette_feather=params['vignette_feather'],
                vignette_roundness=params['vignette_roundness'],
                verbose=True
            )
                
            processed_8bit = (processed * 255).astype(np.uint8)
            processed_bgr = cv2.cvtColor(processed_8bit, cv2.COLOR_RGB2BGR)
                
                
            # Save preview only (25% size)
            preview = cv2.resize(processed_bgr, (0,0), fx=0.25, fy=0.25)
            preview_output_path = output_dir / f"{raw_file.stem}_prev.jpg"
            cv2.imwrite(str(preview_output_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
            if not preview_only:
                output_path = output_dir / f"{raw_file.stem}.jpg"
                cv2.imwrite(str(output_path), processed_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            results.append({
                'combination': combo_dict,
                'output_dir': str(output_dir)
            })
    
    # Save metadata
    metadata = {
        'sweep_type': 'multi_parameter',
        'parameters': param_names,
        'param_grids': param_grids,
        'baseline_params': baseline,
        'total_combinations': total_combinations,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_path = output_base_dir / 'multi_param' / 'sweep_metadata.json'
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ MULTI-PARAMETER SWEEP COMPLETE")
    print("="*70)
    print(f"Results in: {output_base_dir / 'multi_param'}")
    print("="*70)


# ==================== MULTIPROCESSED VERSION ====================

def _process_single_combination_worker(combo_data: Tuple) -> Dict[str, Any]:
    """
    Worker function for parallel processing of a single combination.
    
    This function is called by each worker process in the pool.
    """
    (combo_idx, combo, param_names, preprocessed_image, baseline_params, 
     output_base_dir, raw_file_stem, preview_only) = combo_data
    
    # Create parameter dict
    params = baseline_params.copy()
    combo_dict = dict(zip(param_names, combo))
    params.update(combo_dict)
    
    # Create output directory
    folder_name = "_".join([f"{name}_{combo_dict[name]}" for name in param_names])
    output_dir = output_base_dir / 'multi_param' / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process image
    processed = process_image_complete(
        image=preprocessed_image,
        exposure_stops=params['exposure_stops'],
        highlight_protection=params['highlight_protection'],
        highlight_threshold=params['highlight_threshold'],
        highlight_smoothness=params['highlight_smoothness'],
        highlight_strength=params['highlight_strength'],
        shadow_compression=params['shadow_compression'],
        shadow_threshold=params['shadow_threshold'],
        brightness=params['brightness'],
        contrast=params['contrast'],
        saturation=params['saturation'],
        gamma=params['gamma'],
        vignette_strength=params['vignette_strength'],
        vignette_feather=params['vignette_feather'],
        vignette_roundness=params['vignette_roundness'],
        verbose=False
    )
    
    # Save
    processed_8bit = (processed * 255).astype(np.uint8)
    processed_bgr = cv2.cvtColor(processed_8bit, cv2.COLOR_RGB2BGR)
    
    # Save preview
    preview = cv2.resize(processed_bgr, (0, 0), fx=0.25, fy=0.25)
    preview_output_path = output_dir / f"{raw_file_stem}_prev.jpg"
    cv2.imwrite(str(preview_output_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    # Save full if requested
    if not preview_only:
        output_path = output_dir / f"{raw_file_stem}.jpg"
        cv2.imwrite(str(output_path), processed_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    return {
        'combination': combo_dict,
        'output_dir': str(output_dir),
        'combo_idx': combo_idx
    }


def sweep_multiple_parameters_parallel(
    param_names: List[str],
    color_matrix_path: Path,
    input_dir: Path,
    output_base_dir: Path,
    n_images: int = 2,
    preview_only: bool = True,
    max_workers: int = None
):
    """
    Sweep multiple parameters using concurrent.futures.ProcessPoolExecutor.
    
    Modern, clean implementation with REAL-TIME progress tracking!
    Uses all available CPU cores by default.
    
    Parameters:
        param_names: List of parameter names to sweep
        color_matrix_path: Path to calibration matrix
        input_dir: Directory with RAW files
        output_base_dir: Base directory for outputs
        n_images: Number of images to process
        preview_only: If True, only save 25% preview images
        max_workers: Number of parallel workers (None = use all CPUs)
    """
    
    print("\n" + "="*70)
    print(f"MULTI-PARAMETER SWEEP (PARALLEL)")
    print("="*70)
    
    # Determine number of processes
    if max_workers is None:
        max_workers = os.cpu_count()
    
    print(f"Using {max_workers} parallel workers (CPU cores)")
    
    # Validate parameters
    for param in param_names:
        if param not in PARAMETER_GRIDS:
            raise ValueError(f"Unknown parameter: {param}")
    
    # Get parameter grids
    param_grids = {name: PARAMETER_GRIDS[name]['values'] for name in param_names}
    
    # Calculate total combinations
    total_combinations = 1
    for values in param_grids.values():
        total_combinations *= len(values)
    
    print(f"\nParameters to sweep:")
    for name, values in param_grids.items():
        print(f"  - {name}: {values} ({len(values)} values)")
    
    print(f"\nTotal combinations: {total_combinations}")
    print(f"Processing {n_images} images")
    print(f"Total outputs: {total_combinations * n_images}")
    print(f"Expected speedup: ~{max_workers}x faster than sequential")
    
    if total_combinations > 100:
        print("\n⚠️  WARNING: This will create many combinations!")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Sweep cancelled.")
            return
    
    print("="*70)
    
    # Load resources
    color_matrix = np.load(str(color_matrix_path))
    baseline = get_baseline_params()
    
    # Get RAW files
    raw_files = sorted(list(input_dir.glob('*.RAW')))
    if len(raw_files) == 0:
        print("ERROR: No RAW files found!")
        return
    
    # Limit to n_images
    raw_files = raw_files[:n_images]
    
    print(f"\nProcessing {len(raw_files)} images:")
    for f in raw_files:
        print(f"  - {f.name}")
    
    # Generate all combinations
    param_values = [param_grids[name] for name in param_names]
    combinations = list(product(*param_values))
    
    print(f"\nStarting parallel processing...")
    start_time = datetime.now()
    
    all_results = []
    
    # Process each image
    for img_idx, raw_file in enumerate(raw_files, 1):
        print(f"\n[Image {img_idx}/{len(raw_files)}] {raw_file.name}")
        print("-" * 70)
        
        # Preprocess image (once per image)
        print("  Preprocessing image (black level + demosaic + color correction)...")
        
        # Load RAW
        nparray = load_raw_image(raw_file)
        
        # Black level
        image = nparray.copy()
        if baseline['black_level'] != 0:
            image = apply_black_level(image, baseline['black_level'])
        
        # Demosaic
        demosaiced = demosaic_image(image)
        
        # Color correction
        preprocessed = apply_color_correction(demosaiced, color_matrix)
        
        print(f"  Processing {len(combinations)} combinations in parallel...")
        print(f"  Real-time progress:")
        
        # Use ProcessPoolExecutor for clean parallel processing with real-time progress
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and get futures
            future_to_combo = {
                executor.submit(
                    _process_single_combination_worker,
                    (i, combo, param_names, preprocessed.copy(), baseline,
                     output_base_dir, raw_file.stem, preview_only)
                ): (i, combo) for i, combo in enumerate(combinations, 1)
            }
            
            # Process results as they complete (REAL-TIME progress!)
            completed = 0
            for future in as_completed(future_to_combo):
                combo_idx, combo = future_to_combo[future]
                
                try:
                    result = future.result()
                    all_results.append(result)
                    completed += 1
                    
                    # Progress update every 10% or every 50 combinations
                    if completed % max(1, len(combinations) // 10) == 0 or completed % 50 == 0:
                        percent = (completed / len(combinations)) * 100
                        print(f"    [{completed}/{len(combinations)}] {percent:.1f}% complete")
                    
                except Exception as e:
                    print(f"    ✗ Error processing combination {combo_idx}: {e}")
        
        print(f"  ✓ Completed {len(combinations)} combinations")
    
    # Calculate timing
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print("\n" + "="*70)
    print("✓ PARALLEL SWEEP COMPLETE")
    print("="*70)
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Processed {len(all_results)} image+combination pairs")
    print(f"Average: {elapsed/len(all_results):.2f} sec per combination")
    print(f"Results in: {output_base_dir / 'multi_param'}")
    
    # Save metadata
    metadata = {
        'sweep_type': 'multi_parameter_parallel',
        'parameters': param_names,
        'param_grids': param_grids,
        'baseline_params': baseline,
        'total_combinations': total_combinations,
        'n_images': len(raw_files),
        'max_workers': max_workers,
        'elapsed_seconds': elapsed,
        'results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_path = output_base_dir / 'multi_param' / 'sweep_metadata_parallel.json'
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved: {metadata_path.name}")
    print("="*70)


def sweep_around_baseline(
    color_matrix_path: Path,
    input_dir: Path,
    output_base_dir: Path,
    n_images: int = 3,
    variation_percent: float = 20.0
):
    """
    Sweep parameters around your baseline values.
    
    For each parameter, tests: baseline ± variation_percent
    This helps fine-tune your current settings.
    """
    
    print("\n" + "="*70)
    print(f"BASELINE VARIATION SWEEP (±{variation_percent}%)")
    print("="*70)
    
    baseline = get_baseline_params()
    
    print("Baseline parameters:")
    for key, value in baseline.items():
        print(f"  {key}: {value}")
    
    # Define which parameters to vary
    params_to_vary = [
        'black_level', 'exposure_stops', 'shadow_compression',
        'brightness', 'contrast', 'saturation', 'gamma'
    ]
    
    print(f"\nTesting ±{variation_percent}% variations of:")
    for param in params_to_vary:
        print(f"  - {param}")
    
    # Generate variations
    sweep_configs = [{'name': 'baseline', 'params': baseline.copy()}]
    
    for param in params_to_vary:
        base_value = baseline[param]
        
        # Calculate variations
        if base_value != 0:
            variation = abs(base_value * variation_percent / 100.0)
            lower = base_value - variation
            upper = base_value + variation
        else:
            # For zero values, use small absolute variations
            variation = 0.1
            lower = -variation
            upper = variation
        
        # Lower variation
        params_lower = baseline.copy()
        params_lower[param] = lower
        sweep_configs.append({
            'name': f'{param}_lower',
            'params': params_lower,
            'description': f'{param}={lower:.2f} ({-variation_percent}%)'
        })
        
        # Upper variation
        params_upper = baseline.copy()
        params_upper[param] = upper
        sweep_configs.append({
            'name': f'{param}_upper',
            'params': params_upper,
            'description': f'{param}={upper:.2f} (+{variation_percent}%)'
        })
    
    print(f"\nTotal configurations: {len(sweep_configs)}")
    print(f"Processing {n_images} images per configuration")
    print("="*70)
    
    # Load resources
    color_matrix = np.load(str(color_matrix_path))
    raw_files = sorted(list(input_dir.glob('*.RAW')))[1:n_images+1]
    
    # Process each configuration
    for i, config in enumerate(sweep_configs, 1):
        print(f"\n[{i}/{len(sweep_configs)}] {config['name']}")
        if 'description' in config:
            print(f"  {config['description']}")
        
        params = config['params']
        output_dir = output_base_dir / 'baseline_variation' / config['name']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for raw_file in raw_files:
            nparray = load_raw_image(raw_file)
            demosaiced = demosaic_image(nparray, black_level=params['black_level'])
            
            processed = process_image_complete(
                raw_array=demosaiced,
                color_matrix=color_matrix,
                black_level=0,
                exposure_stops=params['exposure_stops'],
                highlight_protection=params['highlight_protection'],
                highlight_threshold=params['highlight_threshold'],
                highlight_smoothness=params['highlight_smoothness'],
                highlight_strength=params['highlight_strength'],
                shadow_compression=params['shadow_compression'],
                shadow_threshold=params['shadow_threshold'],
                brightness=params['brightness'],
                contrast=params['contrast'],
                saturation=params['saturation'],
                gamma=params['gamma'],
                vignette_strength=params['vignette_strength'],
                vignette_feather=params['vignette_feather'],
                vignette_roundness=params['vignette_roundness'],
                verbose=False
            )
            
            processed_8bit = (processed * 255).astype(np.uint8)
            processed_bgr = cv2.cvtColor(processed_8bit, cv2.COLOR_RGB2BGR)
            
            output_path = output_dir / f"{raw_file.stem}.jpg"
            cv2.imwrite(str(output_path), processed_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        print(f"  ✓ Saved to: {output_dir.name}")
    
    print("\n" + "="*70)
    print("✓ BASELINE VARIATION SWEEP COMPLETE")
    print("="*70)
    print(f"Results in: {output_base_dir / 'baseline_variation'}")
    print(f"Compare to 'baseline' folder to see effects of variations")
    print("="*70)


# ==================== MAIN ====================

if __name__ == "__main__":
    """
    Examples of different sweep modes.
    Uncomment the one you want to run.
    """
    
    # Configuration
    color_matrix_path = Path('/home/mkutuga/SemiF-Preprocessing/calibration_results/data/calibration_matrix_20251119_171642.npy')
    input_dir = Path('/home/mkutuga/SemiF-Preprocessing/colorchecker_raw')

    output_base_dir = Path('parameter_sweeps_phase2')
    
    print("\n" + "="*70)
    print("PARAMETER SWEEP OPTIONS")
    print("="*70)
    print("\n1. Single parameter sweep - Test one parameter at a time")
    print("2. Multiple parameter sweep - Test combinations (SEQUENTIAL)")
    print("3. Multiple parameter sweep - Test combinations (PARALLEL - FAST!)")
    print("4. Baseline variation sweep - Fine-tune current settings (±20%)")
    print("5. Custom sweep - Define your own")
    
    choice = input("\nSelect option (1-5): ")
    
    if choice == '1':
        # OPTION 1: Single parameter sweep
        print("\nAvailable parameters:")
        for i, (name, info) in enumerate(PARAMETER_GRIDS.items(), 1):
            print(f"  {i}. {name} - {info['description']}")
        
        param_num = int(input("\nSelect parameter number: "))
        param_name = list(PARAMETER_GRIDS.keys())[param_num - 1]
        
        sweep_single_parameter(
            param_name=param_name,
            color_matrix_path=color_matrix_path,
            input_dir=input_dir,
            output_base_dir=output_base_dir,
            n_images=3
        )
    
    elif choice == '2':
        # OPTION 2: Multiple parameter sweep (SEQUENTIAL)
        print("\nWARNING: This is the SLOW sequential version!")
        print("Consider using option 3 (parallel) instead for much faster processing.")
        print("\nTest combinations of shadow_compression, contrast, saturation")
        
        response = input("\nContinue with sequential? (yes/no): ")
        if response.lower() == 'yes':
            sweep_multiple_parameters(
                param_names=['contrast', 'brightness', 'saturation', 'exposure_stops', 'vignette_strength'],
                color_matrix_path=color_matrix_path,
                input_dir=input_dir,
                output_base_dir=output_base_dir,
                n_images=2,
                preview_only=True
            )
        else:
            print("Cancelled. Try option 3 for parallel processing!")
    
    elif choice == '3':
        # OPTION 3: Multiple parameter sweep (PARALLEL - FAST!)
        print("\n🚀 PARALLEL MODE - Uses all CPU cores for maximum speed!")
        print("\nTesting combinations of:")
        param_names = [
            # 'exposure_stops', 
            # 'gamma',

            # 'shadow_compression',
            # 'contrast', 

            # 'contrast', 
            # 'saturation',
            # 'brightness',

            # 'highlight_threshold',
            # 'highlight_smoothness',
            # 'highlight_strength',

            # 'vignette_strength'
            ]
        for pname in param_names:
            print(f"  - {pname}: {PARAMETER_GRIDS[pname]['values']}")
        
        n_imgs = int(input("\nHow many images to process? (default 2): ") or "2")
        preview = input("Preview only (25% size)? (yes/no, default yes): ") or "yes"
        preview_only = preview.lower() == 'yes'
        
        sweep_multiple_parameters_parallel(
            param_names=param_names,
            color_matrix_path=color_matrix_path,
            input_dir=input_dir,
            output_base_dir=output_base_dir,
            n_images=n_imgs,
            preview_only=preview_only,
            max_workers=None  # Use all CPUs
        )
    
    elif choice == '4':
        # OPTION 4: Baseline variation sweep
        sweep_around_baseline(
            color_matrix_path=color_matrix_path,
            input_dir=input_dir,
            output_base_dir=output_base_dir,
            n_images=3,
            variation_percent=20.0
        )
    
    elif choice == '5':
        # OPTION 5: Custom sweep - edit as needed
        print("\nEdit the script to define your custom sweep!")
        print("See examples above for how to use the sweep functions.")
    
    else:
        print("Invalid choice!")