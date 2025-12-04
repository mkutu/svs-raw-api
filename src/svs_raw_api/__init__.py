"""
Image Processing Pipeline for Agricultural Computer Vision

A modular pipeline for processing RAW images from industrial cameras,
with a focus on ColorChecker-based calibration, color correction,
and consistent image quality for computer vision applications.

Camera: SVS-Vistek shr661CXGE with Sony IMX661LQA sensor
Lens: inspec.x L 4/60 (60mm, f/4.0)
"""

# Version
__version__ = '1.0.0'
__author__ = 'Matthew Kutugata'
__license__ = 'MIT'

from  .core import SVSRaw2DNG

__all__ = [
    # Version
    '__version__',
    'SVSRaw2DNG'
]