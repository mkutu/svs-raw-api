import numpy as np
from typing import Sequence

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