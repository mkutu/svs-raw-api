"""
SVS RAW Processing Library

Simple library for converting RAW images to DNG and JPG.
Designed for integration into larger CV pipelines.

Usage:
    from svs_raw_api import load_config, RawToDng, DngToJpg
    
    # Load config
    config = load_config("config/svs_shr661.yaml")
    
    # Convert RAW to DNG
    converter = RawToDng(config)
    dng_path = converter.convert("image.raw", "image.dng")
    
    # Develop DNG to JPG
    developer = DngToJpg("/usr/bin/rawtherapee-cli")
    jpg_path = developer.develop(dng_path, "image.jpg")
"""

__version__ = '2.0.0'

from .converter import RawToDng
from .developer import DngToJpg
from .config import load_config

__all__ = ['RawToDng', 'DngToJpg', 'load_config']