"""
RAW to DNG conversion module.

This module handles the conversion of proprietary RAW files to Adobe DNG format
with camera-specific color calibration and metadata.
"""

from pathlib import Path
from typing import Optional, Union
from datetime import datetime, timezone

import numpy as np
from pidng.core import RAW2DNG, DNGTags, Tag


class RawToDng:
    """
    Converts RAW images to DNG format with color calibration.
    
    This class handles the technical details of DNG creation, including:
    - Color matrix application
    - Forward matrix calculation
    - White balance (AsShotNeutral)
    - Camera metadata embedding
    
    Parameters
    ----------
    color_matrix : np.ndarray
        Color calibration matrix (stored as .npy file)
    camera_profile : dict
        Camera-specific configuration (sensor size, black/white levels, etc.)
    
    Examples
    --------
    >>> from svs_raw_api.core import RawConverter
    >>> from svs_raw_api.calibration import CameraProfile
    >>> 
    >>> profile = CameraProfile.load("config/cameras/svs_shr661.yaml")
    >>> converter = RawConverter(profile.color_matrix, profile.camera_config)
    >>> 
    >>> converter.convert(
    ...     raw_path="image.raw",
    ...     output_path="image.dng"
    ... )
    """
    
    def __init__(
        self,
        cfg: dict
    ):
        self.cfg = cfg
        self.color_matrix = cfg['color_matrix']
        self.profile = cfg['dng_tags']
        
        # Pre-compute DNG rational values
        self._prepare_color_matrices()
    
    def _prepare_color_matrices(self) -> None:
        """Convert color matrices to DNG rational format."""
        data = self.color_matrix.item()
        
        # Extract matrices
        cm = data["color_matrix"].T
        fm = data["forward_matrix"].T
        wb = np.asarray(data["wb_gains"], dtype=float)
        
        # Convert to rational format (numerator/denominator pairs)
        matrix_den = 10000
        self.ccm_rational = [
            [int(round(v * matrix_den)), matrix_den] 
            for v in cm.reshape(-1)
        ]
        self.fm_rational = [
            [int(round(v * matrix_den)), matrix_den] 
            for v in fm.reshape(-1)
        ]
        
        # AsShotNeutral = inverse of WB gains
        r_gain, g_gain, b_gain = wb
        self.as_shot_neutral = [
            [int(round(matrix_den / r_gain)), matrix_den],
            [int(round(matrix_den / g_gain)), matrix_den],
            [int(round(matrix_den / b_gain)), matrix_den],
        ]
    
    def _create_dng_tags(self) -> DNGTags:
        # --- DNG TAGS ---
        t = DNGTags()
        # images
        icfg = self.profile['image']
        t.set(Tag.ImageWidth,  icfg['SVCamImageWidth'])
        t.set(Tag.ImageLength, icfg['SVCamImageHeight'])
        t.set(Tag.BitsPerSample, icfg['BitsPerSample'])
        t.set(Tag.PhotometricInterpretation, icfg['PhotometricInterpretation'])
        t.set(Tag.Orientation, icfg['Orientation'])
        t.set(Tag.SamplesPerPixel, icfg['SamplesPerPixel'])
        t.set(Tag.CFARepeatPatternDim, icfg['CFARepeatPatternDim'])
        t.set(Tag.CFAPattern, icfg['CFAPattern'])
        t.set(Tag.RowsPerStrip, icfg['RowsPerStrip'])
        # t.set(Tag.TileWidth,  icfg['TileWidth'])
        # t.set(Tag.TileLength, icfg['TileLength'])

        # Camera
        ccfg = self.profile['camera']
        t.set(Tag.Make,  ccfg['Make'])
        t.set(Tag.Model, ccfg['Model'])
        t.set(Tag.EXIFPhotoBodySerialNumber, ccfg['SerialNumber'])
        t.set(Tag.EXIFPhotoLensModel, ccfg['LensModel'])
        t.set(Tag.FocalLength, [[int(ccfg['FocalLength'] * 10000), 10000]])  # rational
        # t.set(Tag.FocalLengthIn35mmFormat, ccfg.FocalLengthIn35mmFormat)
        t.set(Tag.FocalLengthIn35mmFilm, ccfg['FocalLengthIn35mmFilm'])  # rational
        t.set(Tag.FNumber, [[int(ccfg['FNumber'] * 10000), 10000]])
        t.set(Tag.FocalPlaneXResolution, [[int(ccfg['FocalPlaneXResolution'] * 10000), 10000]])
        t.set(Tag.FocalPlaneYResolution, [[int(ccfg['FocalPlaneYResolution'] * 10000), 10000]])
        t.set(Tag.FocalPlaneResolutionUnit, [ccfg['FocalPlaneResolutionUnit']])
        # t.set(Tag.PixelSize, ccfg['PixelSize'])

        # DNG Core
        dcfg = self.profile['dng']
        t.set(Tag.DNGVersion, dcfg['DNGVersion'])
        t.set(Tag.DNGBackwardVersion, dcfg['DNGBackwardVersion'])
        # 16-bit black and white levels
        t.set(Tag.BlackLevel, dcfg['BlackLevel'])
        t.set(Tag.WhiteLevel, dcfg['WhiteLevel'])

        # Color
        t.set(Tag.ColorMatrix1, self.ccm_rational)
        t.set(Tag.ColorMatrix2, self.ccm_rational)

        t.set(Tag.ForwardMatrix1, self.fm_rational)
        t.set(Tag.ForwardMatrix2, self.fm_rational)

        t.set(Tag.AsShotNeutral, self.as_shot_neutral)

        t.set(Tag.CalibrationIlluminant1, dcfg['CalibrationIlluminant1'])
        t.set(Tag.PreviewColorSpace, dcfg['PreviewColorSpace'])
        t.set(Tag.BaselineExposure, [dcfg['BaselineExposure']])

        return t
    
    @staticmethod
    def _extract_timestamp_from_filename(filename: str) -> str:
        """Extract timestamp from filename and format for EXIF."""
        # Expecting format like: MD_1764960482.raw
        stem = Path(filename).stem
        parts = stem.split('_')
        
        if len(parts) >= 2 and parts[-1].isdigit():
            epoch = int(parts[-1])
            dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
            return dt.strftime("%Y:%m:%d %H:%M:%S")
        
        # Fallback to current time
        return datetime.now(timezone.utc).strftime("%Y:%m:%d %H:%M:%S")
    
    def convert(
        self,
        raw_path: Union[str, Path],
    ) -> Path:
        """
        Convert RAW file to DNG format.
        
        Parameters
        ----------
        raw_path : str or Path
            Path to input RAW file (for metadata extraction)
        output_path : str or Path
            Path where DNG file will be saved
        raw_data : np.ndarray, optional
            Pre-loaded RAW data. If None, will load from raw_path.
        
        Returns
        -------
        Path
            Path to created DNG file
        
        Raises
        ------
        ValueError
            If image dimensions don't match camera profile
        FileNotFoundError
            If raw_path doesn't exist and raw_data not provided
        """
        raw_path = Path(raw_path)
        output_path = Path(self.cfg['paths']['temp_dng_dir']) / raw_path.with_suffix('.dng').name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
            
        expected_size = (
            self.profile['image']['SVCamImageHeight'],
            self.profile['image']['SVCamImageWidth']
        )
        total_pixels = expected_size[0] * expected_size[1]
        
        raw_data = np.fromfile(raw_path, dtype=np.uint16, count=total_pixels)
        raw_data = raw_data.reshape(expected_size)
        
        # Create DNG tags
        tags = self._create_dng_tags()
        
        # Add timestamp tags
        timestamp = self._extract_timestamp_from_filename(raw_path.name)
        tags.set(Tag.DateTimeOriginal, timestamp)
        tags.set(Tag.DateTime, timestamp)
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DNG
        converter = RAW2DNG()
        converter.options(tags, path="", compress=False)
        converter.convert(raw_data, filename=str(output_path))
        
        return output_path