"""
Enhanced Image Processing Pipeline
Incorporates RawTherapee adjustments directly into Python pipeline

This extends the existing pipeline with:
- Black level adjustment
- Shadow compression
- Brightness/contrast adjustments
- Saturation control
- Vignetting effect
"""

import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime
from image_processing_api.constants import WIDTH, HEIGHT, BLACK_LEVEL_SHIFTED, EFFECTIVE_RANGE
from image_processing_api.archive.image_processing import (
    apply_color_correction,
    load_raw_image,
    demosaic_image
)

def print_stats(img, name):
    print(f"{name:30s}: min={np.min(img):.3f}, max={np.max(img):.3f}, "
            f"mean={np.mean(img):.3f}, clipped={np.sum(img >= 0.99)/img.size*100:.2f}%, shape={img.shape}")
# ==================== NEW ADJUSTMENT FUNCTIONS ====================

def apply_black_level(image: np.ndarray, black_value: float) -> np.ndarray:
    """
    Apply black level adjustment (subtract and rescale).
    
    This should be done BEFORE color correction for proper results.
    
    Parameters:
        image: Linear image (0-1 range)
        black_value: Black point (0-255 scale from RawTherapee)
                     Typical: 0-100. Higher values lift shadows.
    
    Returns:
        Black-adjusted image
    """
    # Convert black_value from 0-255 scale to 0-1 scale
    black_normalized = black_value / 255.0
    
    # Subtract black level and rescale
    # Formula: (image - black) / (1 - black)
    adjusted = np.clip((image - black_normalized) / (1.0 - black_normalized), 0, None)
    
    return adjusted


def apply_brightness_contrast(image: np.ndarray, 
                              brightness: float = 0, 
                              contrast: float = 0) -> np.ndarray:
    """
    Apply brightness and contrast adjustments.
    
    Parameters:
        image: Linear or gamma-corrected image (0-1+ range)
        brightness: Brightness adjustment (-100 to +100, typical -50 to +50)
        contrast: Contrast adjustment (-100 to +100, typical -50 to +50)
    
    Returns:
        Adjusted image
    """
    # Convert brightness from RawTherapee scale to multiplier
    # RawTherapee: -100 to +100 scale
    brightness_factor = 1.0 + (brightness / 100.0)
    
    # Apply brightness (simple multiplication)
    result = image * brightness_factor
    
    # Apply contrast adjustment if specified
    if contrast != 0:
        # Contrast as S-curve around midpoint (0.5)
        # Positive contrast increases separation from 0.5
        # Negative contrast decreases separation from 0.5
        contrast_factor = (contrast / 100.0) * 0.5 + 1.0
        
        # Apply contrast: (x - 0.5) * factor + 0.5
        result = (result - 0.5) * contrast_factor + 0.5
    
    return result


def apply_saturation_adjustment(image: np.ndarray, saturation_delta: float = 0) -> np.ndarray:
    """
    Apply saturation adjustment in HSV space.
    
    Parameters:
        image: RGB image (0-1 range, should be gamma-corrected for proper results)
        saturation_delta: Saturation adjustment (-100 to +100)
                         Positive = more saturated, negative = less saturated
                         RawTherapee default: 0, typical range: -50 to +50
    
    Returns:
        Saturation-adjusted RGB image
    """
    if saturation_delta == 0:
        return image
    
    # Clip to valid range for HSV conversion
    image_clipped = np.clip(image, 0, 1)
    
    # Convert to HSV
    # OpenCV expects uint8, so convert
    image_uint8 = (image_clipped * 255).astype(np.uint8)
    hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Adjust saturation channel
    # RawTherapee scale: -100 to +100
    # Convert to multiplier: saturation * (1 + delta/100)
    saturation_factor = 1.0 + (saturation_delta / 100.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    
    # Convert back to RGB
    hsv_uint8 = hsv.astype(np.uint8)
    rgb_uint8 = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2RGB)
    
    return rgb_uint8.astype(np.float32) / 255.0


def apply_shadow_compression(image: np.ndarray, 
                             compression: float = 0,
                             threshold: float = 0.15) -> np.ndarray:
    """
    Apply shadow compression (lifts dark areas).
    
    This is the complement to highlight rolloff - it compresses shadows
    by lifting them in a smooth way.
    
    Parameters:
        image: Linear or gamma-corrected image (0-1 range)
        compression: Amount of compression (0-100)
                    0 = no compression, 100 = maximum lift
        threshold: Shadow threshold (0-1, typically 0.1-0.2)
                  Values below this are considered "shadows"
    
    Returns:
        Shadow-compressed image
    """
    if compression == 0:
        return image
    
    result = image.copy()
    
    # Find pixels below threshold
    mask = image < threshold
    
    if not np.any(mask):
        return result
    
    # Convert compression to strength (0-100 -> 0-1)
    strength = compression / 100.0
    
    # Amount below threshold
    below_threshold = threshold - image[mask]
    
    # Smooth compression - lift shadows
    # Formula: shadow + lift_amount * (1 - exp(-distance_from_threshold))
    lift_amount = threshold * strength
    compressed_amount = lift_amount * (1.0 - np.exp(-below_threshold / (threshold * 0.5)))
    
    # New value = original + lift
    result[mask] = image[mask] + compressed_amount
    
    return result


def apply_vignetting(image: np.ndarray, 
                    strength: float = 0,
                    feather: float = 50,
                    roundness: float = 0) -> np.ndarray:
    """
    Apply vignetting effect (darkening/lightening at edges).
    
    Parameters:
        image: RGB image (0-1 range)
        strength: Vignetting strength (-2.0 to +2.0)
                 Negative = darken edges (typical vignette)
                 Positive = lighten edges
                 RawTherapee scale: typically -2.0 to +2.0
        feather: Transition smoothness (0-100, higher = more gradual)
        roundness: Shape of vignette (-100 to +100)
                  0 = circular, negative = more rectangular
    
    Returns:
        Vignetted image
    """
    if strength == 0:
        return image
    
    h, w = image.shape[:2]
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    
    # Normalize coordinates to [-1, 1]
    x_norm = (x - w/2) / (w/2)
    y_norm = (y - h/2) / (h/2)
    
    # Apply roundness (elliptical shape)
    aspect_ratio = h / w
    if roundness < 0:
        # More rectangular
        y_norm = y_norm * (1 + abs(roundness) / 100.0)
    
    # Calculate distance from center
    distance = np.sqrt(x_norm**2 + y_norm**2)
    
    # Create vignette mask with feathering
    # Feather controls the falloff rate
    feather_factor = 2.0 / (feather / 100.0 + 0.1)
    vignette_mask = 1.0 - np.clip(distance ** feather_factor, 0, 1)
    
    # Apply strength
    # Positive strength lightens edges, negative darkens
    vignette_effect = 1.0 + strength * (1.0 - vignette_mask)
    
    # Apply to image
    result = image * vignette_effect[:, :, np.newaxis]
    
    return np.clip(result, 0, 1)


# ==================== COMPLETE PROCESSING PIPELINE ====================

def process_image_complete(
    image: np.ndarray,
    # Exposure and tone
    exposure_stops: float = 0.0,
    highlight_protection: str = 'rolloff',
    highlight_threshold: float = 0.85,
    highlight_smoothness: float = 0.10,
    highlight_strength: float = 1.0,
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
    # Debugging
    verbose: bool = False
) -> np.ndarray:
    """
    Complete image processing pipeline with all RawTherapee adjustments.
    
    Processing order (critical for proper results):
    1. Demosaic (assumed already done to raw_array)
    2. Black level subtraction (linear)
    3. Color correction (linear)
    4. Exposure compensation (linear)
    5. Highlight protection / tone mapping (linear -> compressed)
    6. Shadow compression (compressed)
    7. Brightness/contrast (any)
    8. Gamma correction (display encoding)
    9. Saturation adjustment (gamma-corrected)
    10. Vignetting (final touch)
    
    Parameters:
        raw_array: Demosaiced image (0-1 range, linear light)
        color_matrix: 3x3 color correction matrix
        
        # Preprocessing
        black_level: Black point (0-255 scale), typical 0-100
        
        # Exposure and tone
        exposure_stops: EV adjustment, typical -1 to +3
        highlight_protection: 'rolloff', 'reinhard', 'aces', 'none'
        highlight_threshold: Start of highlight rolloff, typical 0.8-0.95
        highlight_smoothness: Rolloff smoothness, typical 0.05-0.20
        shadow_compression: Shadow lift amount (0-100), typical 0-80
        shadow_threshold: Shadow region threshold, typical 0.1-0.2
        
        # Creative adjustments
        brightness: Brightness adjustment (-100 to +100), typical -50 to +50
        contrast: Contrast adjustment (-100 to +100), typical -50 to +50
        saturation: Saturation adjustment (-100 to +100), typical -50 to +50
        
        # Display encoding
        gamma: Gamma for display encoding, typical 0.7-1.0 for linear displays
        
        # Post-processing
        vignette_strength: Vignetting (-2 to +2), negative darkens edges
        vignette_feather: Vignette smoothness (0-100)
        vignette_roundness: Vignette shape (-100 to +100)
    
    Returns:
        Processed RGB image (0-1 range)
    """
    
    if verbose:
        def print_stats(img, name):
            print(f"{name:30s}: min={np.min(img):.3f}, max={np.max(img):.3f}, "
                  f"mean={np.mean(img):.3f}, clipped={np.sum(img >= 0.99)/img.size*100:.2f}%, shape={img.shape}")
        
    else:
        print_stats = lambda x, y: None
    
    # 3. Exposure compensation (linear light)
    if exposure_stops != 0:
        image = apply_exposure_compensation(image, exposure_stops)
        print_stats(image, "4. After exposure")
    
    # 4. Highlight protection / tone mapping (linear -> compressed highlights)
    if highlight_protection == 'rolloff':
        image = apply_highlight_rolloff(image, highlight_threshold, highlight_smoothness, strength=highlight_strength)
        print_stats(image, "5. After highlight rolloff")
    elif highlight_protection == 'reinhard':
        image = apply_tone_curve(image, method='reinhard')
        print_stats(image, "5. After reinhard")
    elif highlight_protection == 'aces':
        image = apply_tone_curve(image, method='aces')
        print_stats(image, "5. After ACES")
    else:
        image = np.clip(image, 0, 1)
        print_stats(image, "5. After clipping")
    
    # 5. Shadow compression (lift dark areas)
    if shadow_compression > 0:
        image = apply_shadow_compression(image, shadow_compression, shadow_threshold)
        print_stats(image, "6. After shadow compression")
    
    # 6. Brightness/contrast adjustments
    if brightness != 0 or contrast != 0:
        image = apply_brightness_contrast(image, brightness, contrast)
        print_stats(image, "7. After brightness/contrast")
    
    # 7. Gamma correction (display encoding)
    if gamma != 1.0:
        image = apply_gamma_correction(image, gamma)
        print_stats(image, "8. After gamma")
    
    # 8. Saturation adjustment (in gamma-corrected space)
    if saturation != 0:
        image = apply_saturation_adjustment(image, saturation)
        print_stats(image, "9. After saturation")
    
    # 9. Vignetting (final touch)
    if vignette_strength != 0:
        image = apply_vignetting(image, vignette_strength, vignette_feather, vignette_roundness)
        print_stats(image, "10. After vignetting")
    
    # Final clip to valid range
    image = np.clip(image, 0, 1)
    
    if verbose:
        print()
    
    return image


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction WITHOUT clipping."""
    return np.power(np.clip(image, 0, None), gamma)


def apply_exposure_compensation(image: np.ndarray, stops: float = 0.0) -> np.ndarray:
    """Apply exposure compensation WITHOUT clipping."""
    factor = 2.0 ** stops
    return image * factor


def apply_tone_curve(image: np.ndarray, method: str = 'reinhard') -> np.ndarray:
    """Apply tone mapping."""
    if method == 'reinhard':
        return image / (1.0 + image)
    elif method == 'aces':
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return np.clip((image * (a * image + b)) / (image * (c * image + d) + e), 0, 1)
    else:
        return np.clip(image, 0, 1)


def apply_highlight_rolloff(
    image: np.ndarray, 
    threshold: float = 0.85, 
    smoothness: float = 0.10,
    strength: float = 1.0  # NEW!
) -> np.ndarray:
    """
    Apply smooth highlight rolloff with adjustable strength.
    
    Parameters:
        image: RGB image (0-1+ range)
        threshold: Where to start rolling off (0.70-0.95)
        smoothness: How gradual the rolloff is (0.03-0.20)
        strength: Compression power (0.5=gentle, 1.0=normal, 2.0=aggressive)
    """
    result = image.copy()
    mask = image > threshold
    
    if not np.any(mask):
        return result
    
    above_threshold = image[mask] - threshold
    max_range = 1.0 - threshold
    
    # Apply strength multiplier to compression
    compressed_amount = max_range * (1.0 - np.exp(-strength * above_threshold / smoothness))
    
    result[mask] = threshold + compressed_amount
    return result


# ==================== EXAMPLE USAGE ====================

def example_rawtherapee_profile():
    """
    Example showing how to use parameters from your RawTherapee profile.
    
    From your profile:
    - Exposure compensation: +2.72
    - Black: 93
    - Highlight compression: 100
    - Shadow compression: 53
    - Brightness: 5
    - Contrast: 20
    - Saturation: 15
    - Vignetting: -1.1 strength, 88 feather
    """
    
    # These are the parameters from your RawTherapee profile
    params = {
        'black_level': 93,                    # From Black=93
        'exposure_stops': 2.72,               # From Compensation=2.72
        'highlight_protection': 'rolloff',    # You're using rolloff
        'highlight_threshold': 0.85,          # Good default for agricultural images
        'highlight_smoothness': 0.10,         # Good default
        'shadow_compression': 53,             # From ShadowCompr=53
        'shadow_threshold': 0.15,             # Good default (can tune)
        'brightness': 5,                      # From Brightness=5
        'contrast': 20,                       # From Contrast=20
        'saturation': 15,                     # From Saturation=15
        'gamma': 0.9,                         # From your sweep results
        'vignette_strength': -1.1,            # From PCVignette Strength=-1.1
        'vignette_feather': 88,               # From PCVignette Feather=88
        'vignette_roundness': 0,              # From PCVignette Roundness=0
        'verbose': True
    }
    
    return params


# if __name__ == "__main__":
#     print("Enhanced Image Processing Pipeline")
#     print("="*70)
#     print("\nThis script provides functions to incorporate RawTherapee adjustments")
#     print("directly into your Python pipeline.")
#     print("\nKey functions:")
#     print("  - apply_black_level()")
#     print("  - apply_shadow_compression()")
#     print("  - apply_brightness_contrast()")
#     print("  - apply_saturation_adjustment()")
#     print("  - apply_vignetting()")
#     print("  - process_image_complete()  [main pipeline]")
#     print("\nSee example_rawtherapee_profile() for your specific settings.")
#     print("="*70)
    
#     # Show your RawTherapee parameters
#     params = example_rawtherapee_profile()
#     print("\nYour RawTherapee Parameters:")
#     print("-" * 70)
#     for key, value in params.items():
#         print(f"  {key:25s}: {value}")
