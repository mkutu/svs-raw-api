#!/usr/bin/env python3
"""
Compute DNG ColorMatrix / ForwardMatrix from ColorChecker calibration JSON
and report color error statistics.

Input JSON must contain:
- "measured_colors": 24x3 list of camera-linear RGB
    (after black-level subtract, normalized, demosaiced, NO white balance)
- "reference_colors": 24x3 list of sRGB patch values in [0, 1]
    (standard ColorChecker 24)

Output:
- 3x3 ForwardMatrix (camera -> XYZ, D65)
- 3x3 ColorMatrix  (XYZ -> camera)
- Per-patch ΔE*ab statistics between calibrated result and reference

Matrices are printed as space-separated floats suitable for DNG/ExifTool tags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from svs_raw_api.constants import M_SRGB_TO_XYZ


import json
import numpy as np
from pathlib import Path

def compute_wb(measured: np.ndarray,
                neutral_indices=None) -> list[float]:
    """
    Compute AsShotNeutral from calibration JSON.

    Parameters
    ----------
    json_path : Path
        Path to JSON with "measured_colors" and "reference_colors".
    neutral_indices : list[int] | None
        Indices of neutral patches to use (0-based, length >= 1).
        If None, defaults to the last 6 patches (CC24 neutrals).

    Returns
    -------
    list[float]
        [R_cam/G_cam, 1.0, B_cam/G_cam]
        suitable for DNG Tag.AsShotNeutral.
    """
    if neutral_indices is None:
        # For CC24, neutrals are patches 18–23 (0-based)
        neutral_indices = list(range(18, 24))

    neutrals = measured[neutral_indices, :]
    mean_r, mean_g, mean_b = neutrals.mean(axis=0)

    r_gain = mean_g / mean_r if mean_r > 0 else 1.0
    g_gain = 1.0
    b_gain = mean_g / mean_b if mean_b > 0 else 1.0

    return [float(r_gain), float(g_gain), float(b_gain)]

# ---------- Color space helpers ----------

def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """
    Convert gamma-encoded sRGB [0, 1] to linear sRGB.

    Parameters
    ----------
    srgb : np.ndarray
        Array of shape (..., 3) with sRGB values in [0, 1].

    Returns
    -------
    np.ndarray
        Same shape as input, linearized.
    """
    srgb = np.asarray(srgb, dtype=float)
    out = np.empty_like(srgb)

    threshold = 0.04045
    low = srgb <= threshold
    high = ~low

    out[low] = srgb[low] / 12.92
    out[high] = ((srgb[high] + 0.055) / 1.055) ** 2.4
    return out


def srgb_to_xyz_d65(linear_srgb: np.ndarray) -> np.ndarray:
    """
    Convert linear sRGB to XYZ (D65).

    Parameters
    ----------
    linear_srgb : np.ndarray
        Array of shape (..., 3) in linear sRGB.

    Returns
    -------
    np.ndarray
        Array of shape (..., 3) in XYZ (D65).
    """
    return linear_srgb @ M_SRGB_TO_XYZ.T


def xyz_to_lab(xyz: np.ndarray, white: Tuple[float, float, float] = (0.95047, 1.0, 1.08883)) -> np.ndarray:
    """
    Convert XYZ to CIE L*a*b* (D65).

    Parameters
    ----------
    xyz : np.ndarray
        Array of shape (..., 3) in XYZ.
    white : tuple
        Reference white (Xn, Yn, Zn). Default is D65.

    Returns
    -------
    np.ndarray
        Array of shape (..., 3) in Lab.
    """
    xyz = np.asarray(xyz, dtype=float)
    Xn, Yn, Zn = white

    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn

    eps = (6 / 29) ** 3
    kappa = 903.3

    def f(t):
        t = np.asarray(t)
        higher = t > eps
        ft = np.empty_like(t)
        ft[higher] = np.cbrt(t[higher])
        ft[~higher] = (kappa * t[~higher] + 16) / 116.0
        return ft

    fx = f(x)
    fy = f(y)
    fz = f(z)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.stack([L, a, b], axis=-1)


def delta_e_lab(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    Compute simple ΔE*ab (Euclidean distance in Lab space).

    Parameters
    ----------
    lab1 : np.ndarray
        Array of shape (..., 3) in Lab.
    lab2 : np.ndarray
        Array of shape (..., 3) in Lab.

    Returns
    -------
    np.ndarray
        ΔE for each corresponding pair.
    """
    diff = lab1 - lab2
    return np.sqrt(np.sum(diff ** 2, axis=-1))


# ---------- Calibration helpers ----------

def load_calibration_json(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load measured and reference colors from calibration JSON.

    Parameters
    ----------
    path : Path
        Path to JSON file.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        measured_colors (24x3), reference_colors (24x3) as float arrays.
    """
    with path.open("r") as f:
        data = json.load(f)

    measured = np.asarray(data["measured_colors"], dtype=float)
    reference_srgb = np.asarray(data["reference_colors"], dtype=float)

    if measured.shape != (24, 3) or reference_srgb.shape != (24, 3):
        raise ValueError(
            f"Expected (24,3) arrays, got "
            f"measured={measured.shape}, reference={reference_srgb.shape}"
        )

    return measured, reference_srgb


def compute_forward_matrix(measured_rgb: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    """
    Solve for ForwardMatrix (camera RGB -> XYZ, D65) using least squares.

    We find F such that:
        measured_rgb @ F ≈ target_xyz

    Parameters
    ----------
    measured_rgb : np.ndarray
        24x3 camera-linear RGB values.
    target_xyz : np.ndarray
        24x3 target XYZ values (from reference patches).

    Returns
    -------
    np.ndarray
        3x3 ForwardMatrix.
    """
    F, _, _, _ = np.linalg.lstsq(measured_rgb, target_xyz, rcond=None)
    return F


def compute_color_matrix(forward_matrix: np.ndarray) -> np.ndarray:
    """
    Compute ColorMatrix (XYZ -> camera RGB) as the inverse of ForwardMatrix.

    Parameters
    ----------
    forward_matrix : np.ndarray
        3x3 ForwardMatrix.

    Returns
    -------
    np.ndarray
        3x3 ColorMatrix.
    """
    return np.linalg.inv(forward_matrix)


def format_for_dng(matrix: np.ndarray, transpose: bool = True, decimals: int = 6) -> str:
    """
    Format a 3x3 matrix as a space-separated string for DNG tags.

    Parameters
    ----------
    matrix : np.ndarray
        3x3 matrix.
    transpose : bool, optional
        If True, transpose before flattening. This matches the
        orientation used in your working DNG.
    decimals : int, optional
        Number of decimal places to round to.

    Returns
    -------
    str
        Space-separated float string.
    """
    if transpose:
        matrix = matrix.T

    flat = matrix.flatten()
    rounded = [round(float(x), decimals) for x in flat]
    return " ".join(f"{v:.{decimals}f}" for v in rounded)


def compute_error_stats(measured_rgb: np.ndarray, reference_xyz: np.ndarray, forward_matrix: np.ndarray) -> None:
    """
    Compute and print ΔE*ab statistics between calibrated result and reference.

    Parameters
    ----------
    measured_rgb : np.ndarray
        24x3 camera-linear RGB.
    reference_xyz : np.ndarray
        24x3 target XYZ (from reference patches).
    forward_matrix : np.ndarray
        3x3 ForwardMatrix (camera -> XYZ).
    """
    # Predicted XYZ from camera + forward matrix
    pred_xyz = measured_rgb @ forward_matrix

    # Convert both to Lab for ΔE
    ref_lab = xyz_to_lab(reference_xyz)
    pred_lab = xyz_to_lab(pred_xyz)

    delta_e = delta_e_lab(pred_lab, ref_lab)

    print("\n# Per-patch ΔE*ab (Lab, D65)")
    print("Patch\tΔE*ab")
    for i, de in enumerate(delta_e, start=1):
        print(f"{i:2d}\t{de:.3f}")

    print("\n# ΔE*ab summary")
    print(f"Mean ΔE*ab   : {np.mean(delta_e):.3f}")
    print(f"Median ΔE*ab : {np.median(delta_e):.3f}")
    print(f"Max ΔE*ab    : {np.max(delta_e):.3f}")
    print(f"Min ΔE*ab    : {np.min(delta_e):.3f}")


# ---------- CLI entrypoint ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute DNG ColorMatrix/ForwardMatrix and ΔE stats from calibration JSON."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to calibration JSON file (with measured_colors and reference_colors).",
    )
    args = parser.parse_args()

    measured, reference_srgb = load_calibration_json(args.json_path)

    # 1) Reference sRGB -> linear -> XYZ
    reference_lin = srgb_to_linear(reference_srgb)
    reference_xyz = srgb_to_xyz_d65(reference_lin)

    # 2) Solve for ForwardMatrix (camera -> XYZ)
    F = compute_forward_matrix(measured, reference_xyz)

    # 3) Compute ColorMatrix (XYZ -> camera)
    C = compute_color_matrix(F)

    # 4) Format for DNG tags
    forward_str = format_for_dng(F, transpose=True, decimals=6)
    color_str = format_for_dng(C, transpose=True, decimals=6)

    print("# ForwardMatrix1/2 (camera -> XYZ, D65)")
    print(forward_str)
    print()
    print("# ColorMatrix1/2 (XYZ -> camera)")
    print(color_str)

    # 5) Error stats
    compute_error_stats(measured, reference_xyz, F)


if __name__ == "__main__":
    main()
