# colorcalib/constants.py
from __future__ import annotations

from typing import List
import numpy as np

# ============================================================================
# CAMERA SPECIFICATIONS
# ============================================================================
SENSOR_MODEL: str = "Sony IMX661LQA"
CAMERA_MODEL: str = "SVS-Vistek shr661CXGE"
LENS_MODEL: str = "inspec.x L 4/60"
FOCAL_LENGTH_MM: float = 60.0
IMAGE_CIRCLE_MM: float = 70.0
F_NUMBER: float = 4.0
F_NUMBER_RANGE: tuple = (4, 32)
BIT_DEPTH: int = 12
WIDTH: int = 13_376
HEIGHT: int = 9_528

# ============================================================================
# CAMERA CONSTANTS - Measured from dark frame analysis
# ============================================================================

# Black level - MEASURED from dark frame (lens cap on)
BLACK_LEVEL_SHIFTED: int = 368  # 16-bit value
BLACK_LEVEL_12BIT: int = 23     # Equivalent 12-bit value

# Bit depth
RAW_MAX_VALUE: int = 65520  # 4095 << 4 (12-bit left-shifted to 16-bit)
EFFECTIVE_RANGE: int = RAW_MAX_VALUE - BLACK_LEVEL_SHIFTED  # 65152


PATCH_NAMES: List[str] = [
    "Dark Skin", "Light Skin", "Blue Sky", "Foliage", "Blue Flower", "Bluish Green",
    "Orange", "Purplish Blue", "Moderate Red", "Purple", "Yellow Green", "Orange Yellow",
    "Blue", "Green", "Red", "Yellow", "Magenta", "Cyan",
    "White", "Neutral 8", "Neutral 6.5", "Neutral 5", "Neutral 3.5", "Black",
]

NUMBER_OF_PATCHES: int = len(PATCH_NAMES)

COLORCHECKER_REFERENCE_SRGB: np.ndarray = np.array([
    [115, 82, 68],    # 1. Dark Skin
    [194, 150, 130],  # 2. Light Skin
    [98, 122, 157],   # 3. Blue Sky
    [87, 108, 67],    # 4. Foliage
    [133, 128, 177],  # 5. Blue Flower
    [103, 189, 170],  # 6. Bluish Green
    [214, 126, 44],   # 7. Orange
    [80, 91, 166],    # 8. Purplish Blue
    [193, 90, 99],    # 9. Moderate Red
    [94, 60, 108],    # 10. Purple
    [157, 188, 64],   # 11. Yellow Green
    [224, 163, 46],   # 12. Orange Yellow
    [56, 61, 150],    # 13. Blue
    [70, 148, 73],    # 14. Green
    [175, 54, 60],    # 15. Red
    [231, 199, 31],   # 16. Yellow
    [187, 86, 149],   # 17. Magenta
    [8, 133, 161],    # 18. Cyan
    [243, 243, 243],  # 19. White
    [200, 200, 200],  # 20. Neutral 8
    [160, 160, 160],  # 21. Neutral 6.5
    [122, 122, 121],  # 22. Neutral 5
    [85, 85, 85],     # 23. Neutral 3.5
    [52, 52, 52]      # 24. Black
]) / 255.0