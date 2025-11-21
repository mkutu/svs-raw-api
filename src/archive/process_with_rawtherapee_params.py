"""
Updated Image Processing Script with RawTherapee Adjustments

This replaces the apply_to_all_images() function with an enhanced version
that includes all RawTherapee adjustments directly in the pipeline.
"""

import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

# Import the enhanced processing functions
from image_processing_api.archive.image_process_enhanced import (
    process_image_complete,
    apply_black_level, 
    print_stats
)
from image_processing_api.archive.image_processing import (
    apply_color_correction,
    load_raw_image,
    demosaic_image
)

# Import original functions (you'd import these from your original script)
# For now, we'll assume they're available



def apply_to_all_images_enhanced(
        color_matrix_path: Path,
        input_dir: Path,
        output_dir: Path = Path('experimental/calibrated_output_rawthereapee'),
        preview_only: bool = False,
        # Preprocessing
        black_level: float = 0,
        # Exposure and tone
        exposure_stops: float = 0.0,
        highlight_protection: str = 'rolloff',
        highlight_threshold: float = 0.85,
        highlight_smoothness: float = 0.10,
        shadow_compression: float = 0,
        shadow_threshold: float = 0.15,
        # Creative adjustments
        brightness: float = 0,
        contrast: float = 0,
        saturation: float = 0,
        # Display encoding
        gamma: float = 1.0,
        # Post-processing
        vignette_strength: float = 0,
        vignette_feather: float = 50,
        vignette_roundness: float = 0,
        # Options
        verbose: bool = False
):
    """
    Apply complete processing pipeline to all RAW images.
    
    This is an enhanced version that includes all RawTherapee adjustments.
    
    Parameters from your RawTherapee profile:
        black_level=93
        exposure_stops=2.72
        shadow_compression=53
        brightness=5
        contrast=20
        saturation=15
        gamma=0.9
        vignette_strength=-1.1
        vignette_feather=88
    """
    
    print("\n" + "="*70)
    print("ENHANCED IMAGE PROCESSING PIPELINE")
    print("="*70)
    
    # Load calibration matrix
    if not color_matrix_path.exists():
        print("\nERROR: Color correction matrix not found!")
        print(f"Looking for: {color_matrix_path}")
        return
    
    color_matrix = np.load(str(color_matrix_path))
    print(f"\n✓ Loaded color correction matrix")
    
    # Print all settings
    print(f"\nProcessing Settings:")
    print(f"  {'Parameter':<30} {'Value':<20}")
    print(f"  {'-'*30} {'-'*20}")
    print(f"  {'Black level':<30} {black_level:<20.1f}")
    print(f"  {'Exposure compensation':<30} {exposure_stops:+.2f} stops")
    print(f"  {'Highlight protection':<30} {highlight_protection:<20}")
    if highlight_protection == 'rolloff':
        print(f"    {'- Threshold':<28} {highlight_threshold:<20.2f}")
        print(f"    {'- Smoothness':<28} {highlight_smoothness:<20.2f}")
    print(f"  {'Shadow compression':<30} {shadow_compression:<20.1f}")
    print(f"  {'Brightness':<30} {brightness:+.1f}")
    print(f"  {'Contrast':<30} {contrast:+.1f}")
    print(f"  {'Saturation':<30} {saturation:+.1f}")
    print(f"  {'Gamma':<30} {gamma:<20.2f}")
    if vignette_strength != 0:
        print(f"  {'Vignetting':<30} {vignette_strength:<20.2f}")
        print(f"    {'- Feather':<28} {vignette_feather:<20.1f}")
    
    # Set up directories
    if not input_dir.exists():
        print(f"\nERROR: Input directory not found: {input_dir}")
        return
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find RAW files
    raw_files = sorted(list(input_dir.glob('*.RAW')))
    
    # For testing, process subset (remove this line for production)
    # raw_files = raw_files[1:5]  # Skip first, process next 4
    
    print(f"\nFound {len(raw_files)} RAW files to process")
    
    if len(raw_files) == 0:
        print("No RAW files found!")
        return
    
    # Process each file
    print(f"\nProcessing images...")
    print("="*70)
    
    for i, raw_file in enumerate(raw_files, 1):
        print(f"\n[{i}/{len(raw_files)}] {raw_file.name}")
        print("-" * 70)

        # Load and demosaic RAW image
        np_array = load_raw_image(raw_file)
        print_stats(np_array, "0. Loaded RAW array")
        
        # 1. Black level adjustment (linear light, before color correction)
        image = np_array.copy()
        if black_level != 0:
            image = apply_black_level(image, black_level)
            print_stats(image, "1. After black level")

        image = demosaic_image(image)
        print_stats(image, "2. Input (demosaiced)")
        
        # 2. Color correction (linear light)
        image = apply_color_correction(image, color_matrix)
        print_stats(image, "3. After color correction")
        
        # perform sweep starting here
    
        # Apply complete processing pipeline
        processed = process_image_complete(
            image=image,
            exposure_stops=exposure_stops,
            highlight_protection=highlight_protection,
            highlight_threshold=highlight_threshold,
            highlight_smoothness=highlight_smoothness,
            shadow_compression=shadow_compression,
            shadow_threshold=shadow_threshold,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            gamma=gamma,
            vignette_strength=vignette_strength,
            vignette_feather=vignette_feather,
            vignette_roundness=vignette_roundness,
            verbose=verbose
        )
        
        # Convert to 8-bit BGR for saving
        processed_8bit = (processed * 255).astype(np.uint8)
        processed_bgr = cv2.cvtColor(processed_8bit, cv2.COLOR_RGB2BGR)
        
        # Save
        if preview_only:
            # Save preview only (25% size)
            resized = cv2.resize(processed_bgr, (0, 0), fx=0.25, fy=0.25)
            preview_path = output_dir / f"{raw_file.stem}_preview.jpg"
            cv2.imwrite(str(preview_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"✓ Saved preview: {preview_path.name}")
        else:
            # Save both preview and full resolution
            resized = cv2.resize(processed_bgr, (0, 0), fx=0.25, fy=0.25)
            preview_path = output_dir / f"{raw_file.stem}_preview.jpg"
            cv2.imwrite(str(preview_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            full_path = output_dir / f"{raw_file.stem}.jpg"
            cv2.imwrite(str(full_path), processed_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"✓ Saved: {full_path.name}")
    
    print("\n" + "="*70)
    print("✓ PROCESSING COMPLETE")
    print(f"Processed {len(raw_files)} images")
    print(f"Output: {output_dir.absolute()}")
    print("="*70)



# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    """
    Example: Process images with your RawTherapee settings
    """
    
    # Your RawTherapee profile parameters
    RAWTHERAPEE_PARAMS = {
        'black_level': 0,
        'exposure_stops': 2.72,
        'highlight_protection': 'rolloff',
        'highlight_threshold': 0.85,
        'highlight_smoothness': 0.10,
        'shadow_compression': 53,
        'shadow_threshold': 0.15,
        'brightness': 5,
        'contrast': 20,
        'saturation': 15,
        'gamma': 0.9,
        'vignette_strength': 1.1,
        'vignette_feather': 88,
        'vignette_roundness': 0,
        'verbose': True  # Set True to see per-step stats
    }
    
    # File paths
    color_matrix_path = Path('/home/mkutuga/SemiF-Preprocessing/calibration_results/data/calibration_matrix_20251119_171642.npy')
    input_dir = Path('/home/mkutuga/SemiF-Preprocessing/colorchecker_raw')
    output_dir = Path('rawtherapee_integrated')
    
    # Process images
    apply_to_all_images_enhanced(
        color_matrix_path=color_matrix_path,
        input_dir=input_dir,
        output_dir=output_dir,
        preview_only=True,
        **RAWTHERAPEE_PARAMS
    )
