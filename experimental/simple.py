from pathlib import Path
import numpy as np
from typing import Sequence
from image_processing_api import PATCH_NAMES
from image_processing_api import load_raw_image, demosaic_image, apply_color_correction
import cv2
def compute_white_balance_gains(
    patch_colors: np.ndarray,
    patch_names: Sequence[str],
    grey_patch_names: Sequence[str],
) -> np.ndarray:
    """
    Compute RGB white-balance gains from ColorChecker neutral patches.

    Args:
        patch_colors: (N, 3) array of linear RGB patch means (0–1), in the same
                      order as `patch_names`.
        patch_names:  list/sequence of patch names, length N.
        grey_patch_names: names of patches to use as neutrals.

    Returns:
        gains: np.ndarray shape (3,) with per-channel gains [gain_R, gain_G, gain_B].
               These are meant to be applied in linear space.
    """
    patch_colors = np.asarray(patch_colors, dtype=np.float64)

    # Find indices for neutral patches
    grey_indices = [i for i, name in enumerate(patch_names) if name in grey_patch_names]
    if not grey_indices:
        raise ValueError("No grey patches found in patch_names for white balance.")

    grey_rgb = patch_colors[grey_indices]  # (K, 3)

    # Average across selected neutral patches
    mean_rgb = grey_rgb.mean(axis=0)  # [R_mean, G_mean, B_mean]

    # Anchor to green channel
    G_ref = mean_rgb[1] if mean_rgb[1] > 0 else 1.0
    gains = np.array([
        G_ref / mean_rgb[0] if mean_rgb[0] > 0 else 1.0,
        1.0,
        G_ref / mean_rgb[2] if mean_rgb[2] > 0 else 1.0,
    ], dtype=np.float64)

    return gains

def apply_white_balance(
    image_linear: np.ndarray,
    gains: np.ndarray,
    clip: bool = False,
) -> np.ndarray:
    """
    Apply per-channel white balance gains to a linear RGB image.

    Args:
        image_linear: HxWx3 linear RGB image (float, typically 0–1).
        gains: array-like of length 3 [gain_R, gain_G, gain_B].
        clip: if True, clip result to [0, 1] (otherwise leave as-is; you can
              clip later after CCM/tone mapping).

    Returns:
        White-balanced linear RGB image.
    """
    gains = np.asarray(gains, dtype=np.float64).reshape(1, 1, 3)
    balanced = image_linear * gains

    if clip:
        balanced = np.clip(balanced, 0.0, 1.0)

    return balanced

raw_colorchecker = Path('/home/mkutuga/SemiF-Preprocessing/colorchecker_raw/MD_1759501672.RAW')
colorchecker_matrix = Path('/home/mkutuga/SemiF-Preprocessing/calibration_results/data/calibration_matrix_20251119_171642.npy')
data = Path("/home/mkutuga/SemiF-Preprocessing/calibration_results/data/calibration_data_20251119_171642.npy")
output_dir = Path('/home/mkutuga/SemiF-Preprocessing/colorchecker_raw/results/')
output_dir.mkdir(parents=True, exist_ok=True)

ccm = np.load(colorchecker_matrix, allow_pickle=True)
cal_data = np.load(data, allow_pickle=True).item()

patch_colors = cal_data.get('measured_colors')
grey_patch_names = PATCH_NAMES[19:24]  # White through Black

wb_gains = compute_white_balance_gains(
    patch_colors=patch_colors,
    patch_names=PATCH_NAMES,
    grey_patch_names=grey_patch_names,
)



checker_top_left       = (5258, 5863)
checker_bottom_right   = (6043, 6817)

bayer_linear = load_raw_image(raw_colorchecker)        # [H, W], linear [0,1]
rgb_linear   = demosaic_image(bayer_linear)    # [H, W, 3], linear [0,1]
rgb_wb = apply_white_balance(rgb_linear, wb_gains, clip=True)
rgb_ccm = apply_color_correction(rgb_wb, ccm)



# convert from rgb to bgr for cv2.imwrite
bgr_wb = rgb_wb[:, :, ::-1]
# resize for faster processing
bgr_wb_small = cv2.resize(bgr_wb, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
wb_output_path = output_dir / 'MD_1759501672_wb_preview.png'
cv2.imwrite(str(wb_output_path), (bgr_wb_small * 255).astype(np.uint8))
wb_output_path = output_dir / 'MD_1759501672_wb_fullres.png'
cv2.imwrite(str(wb_output_path), (bgr_wb * 255).astype(np.uint8))
# convert from rgb to bgr for cv2.imwrite
bgr_ccm = rgb_ccm[:, :, ::-1]
# resize for faster processing
bgr_ccm_small = cv2.resize(bgr_ccm, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
# save output
ccm_output_path = output_dir / 'MD_1759501672_ccm_preview.png'
cv2.imwrite(str(ccm_output_path), (bgr_ccm_small * 255).astype(np.uint8))
ccm_output_path = output_dir / 'MD_1759501672_ccm_fullres.png'
cv2.imwrite(str(ccm_output_path), (bgr_ccm * 255).astype(np.uint8))