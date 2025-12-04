"""
Core RAW to DNG conversion functionality.
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import numpy as np
from pidng.core import RAW2DNG, DNGTags, Tag


class SVSRaw2DNG:
    """
    Convert Sony ARW/RAW images to Adobe DNG format with custom color calibration.
    
    This class handles the conversion of raw sensor data to standardized DNG files
    with embedded color matrices and camera metadata optimized for the SVS-Vistek
    shr661CXGE camera system.
    
    Parameters
    ----------
    color_matrix : np.ndarray
        3x3 color correction matrix for camera calibration
    
    Examples
    --------
    >>> import numpy as np
    >>> from svs_raw_api import SVSRaw2DNG
    >>> 
    >>> # Load calibration matrix
    >>> color_matrix = np.load("calibration_matrix.npy")
    >>> 
    >>> # Create converter
    >>> converter = SVSRaw2DNG(color_matrix=color_matrix)
    >>> 
    >>> # Load RAW image
    >>> raw_data = np.fromfile("image.ARW", dtype=np.uint16)
    >>> raw_image = raw_data.reshape((3024, 4032))
    >>> 
    >>> # Convert to DNG
    >>> camera_tags = {"Make": "SVS-Vistek", "Model": "shr661CXGE"}
    >>> converter.save_dng(raw_image, "output.dng", camera_tags)
    """
    
    def __init__(
        self,
        color_matrix: np.ndarray,
    ):
        self.color_matrix = color_matrix
        
        # Convert color matrix to DNG rational format
        self.color_matrix_dng = self.color_profile_to_rational(matrix_den=10000)
        self.ccm_rational = self.color_matrix_dng["ccm_r"]
        self.fm_rational = self.color_matrix_dng["fm_r"]
        self.as_shot_neutral = self.color_matrix_dng["asn"]
    
    @staticmethod
    def calculate_dt_from_epoch_gmt(file_stem: int) -> str:
        epoch_gmt = int(file_stem.split('_')[-1])
        dt = datetime.fromtimestamp(epoch_gmt, tz=timezone.utc)
        return dt.strftime("%Y:%m:%d %H:%M:%S")
    
    def color_profile_to_rational(self, matrix_den: int) -> Dict[str, Any]:
        data = self.color_matrix.item()

        cm = data["color_matrix"].T
        fm = data["forward_matrix"].T
        wb = np.asarray(data["wb_gains"], dtype=float)

        # Convert matrices to DNG rational form
        ccm_r = [[int(round(v * matrix_den)), matrix_den] for v in cm.reshape(-1)]
        fm_r = [[int(round(v * matrix_den)), matrix_den] for v in fm.reshape(-1)]

        # AsShotNeutral = inverse of WB gains
        r_gain, g_gain, b_gain = wb
        asn = [
            [int(round(matrix_den / r_gain)), matrix_den],
            [int(round(matrix_den / g_gain)), matrix_den],
            [int(round(matrix_den / b_gain)), matrix_den],
        ]

        return {
            "ccm_r": ccm_r,
            "fm_r": fm_r,
            "asn": asn,
        }

    def _create_dng_tags(self, camera_tags: Optional[Dict[str, Any]] = None) -> DNGTags:
        # --- DNG TAGS ---
        t = DNGTags()
        # images
        icfg = camera_tags['image']
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
        ccfg = camera_tags['camera']
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
        dcfg = camera_tags['dng']
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
    
    def save_dng(
        self,
        raw_image: np.ndarray,
        output_path: str,
        camera_tags: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Convert raw image data to DNG format and save to file.
        
        Parameters
        ----------
        raw_image : np.ndarray
            Raw sensor data as 2D numpy array (height × width)
        output_path : str
            Output file path for DNG file
        camera_tags : dict, optional
            Additional camera metadata tags
        
        Returns
        -------
        Path
            Path to created DNG file
        
        Raises
        ------
        ValueError
            If raw_image dimensions don't match configured height/width
        """
        # Validate input dimensions
        height = camera_tags['image']['SVCamImageHeight']
        width = camera_tags['image']['SVCamImageWidth']
        if raw_image.shape != (height, width):
            raise ValueError(
                f"Image dimensions {raw_image.shape} don't match "
                f"configured dimensions ({height}, {width})"
            )
        
        # Create DNG tags
        tags = self._create_dng_tags(camera_tags)

        # Final tags having to do with time and file time stamp
        tags.set(Tag.DateTimeOriginal, self.calculate_dt_from_epoch_gmt(Path(output_path).stem))
        tags.set(Tag.DateTime, self.calculate_dt_from_epoch_gmt(Path(output_path).stem))
        
        
        # Convert to DNG using pidng
        converter = RAW2DNG()
        converter.options(tags, path="", compress=False)
        converter.convert(raw_image, filename=str(output_path))
        return output_path
