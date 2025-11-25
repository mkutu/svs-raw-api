from __future__ import annotations
from pathlib import Path
import yaml
from datetime import datetime, timezone

import numpy as np
from pidng.core import RAW2DNG, DNGTags, Tag

from svs_raw_api import (
    # CalibrationConfig,
    WIDTH,
    HEIGHT,
    SVCamTagConfig,
    ColorConfig
)

class SVSRaw2DNG:
    def __init__(self, config_path: Path = None, matrix_path: Path = None):
        self.config_path = config_path
        self.matrix_path = matrix_path
        self._load_svcam_config() 

        color_cfg = ColorConfig(calibration_matrix_path=self.matrix_path)
        self.ccm_rational = color_cfg.color_matrix_dng
        self.fm_rational = color_cfg.forward_matrix_dng
        self.as_shot_neutral = color_cfg.as_shot_neutral_dng
    
    def _load_svcam_config(self) -> SVCamTagConfig:
        cfg = yaml.safe_load(self.config_path.read_text())
        self.cfg = SVCamTagConfig(**cfg)
        
    @staticmethod
    def calculate_dt_from_epoch_gmt(file_stem: int) -> str:
        epoch_gmt = int(file_stem.split('_')[-1])
        dt = datetime.fromtimestamp(epoch_gmt, tz=timezone.utc)
        return dt.strftime("%Y:%m:%d %H:%M:%S")

    def define_tags(self, raw_file: Path):
        # --- DNG TAGS ---
        t = DNGTags()
        # images
        icfg = self.cfg.image
        t.set(Tag.ImageWidth,  icfg.SVCamImageWidth)
        t.set(Tag.ImageLength, icfg.SVCamImageHeight)
        t.set(Tag.BitsPerSample, icfg.BitsPerSample)
        t.set(Tag.PhotometricInterpretation, icfg.PhotometricInterpretation)
        t.set(Tag.Orientation, icfg.Orientation)
        t.set(Tag.SamplesPerPixel, icfg.SamplesPerPixel)
        t.set(Tag.CFARepeatPatternDim, icfg.CFARepeatPatternDim)
        t.set(Tag.CFAPattern, icfg.CFAPattern)
        t.set(Tag.RowsPerStrip, icfg.RowsPerStrip)
        # t.set(Tag.TileWidth,  icfg.TileWidth)
        # t.set(Tag.TileLength, icfg.TileLength)

        # Camera
        ccfg = self.cfg.camera
        t.set(Tag.Make,  ccfg.Make)
        t.set(Tag.Model, ccfg.Model)
        t.set(Tag.EXIFPhotoBodySerialNumber, ccfg.SerialNumber)
        t.set(Tag.EXIFPhotoLensModel, ccfg.LensModel)
        t.set(Tag.FocalLength, [[int(ccfg.FocalLength * 10000), 10000]])  # rational
        # t.set(Tag.FocalLengthIn35mmFormat, ccfg.FocalLengthIn35mmFormat)
        t.set(Tag.FocalLengthIn35mmFilm, ccfg.FocalLengthIn35mmFilm)  # rational
        t.set(Tag.FNumber, [[int(ccfg.FNumber * 10000), 10000]])
        t.set(Tag.FocalPlaneXResolution, [[int(ccfg.FocalPlaneXResolution * 10000), 10000]])
        t.set(Tag.FocalPlaneYResolution, [[int(ccfg.FocalPlaneYResolution * 10000), 10000]])
        t.set(Tag.FocalPlaneResolutionUnit, [ccfg.FocalPlaneResolutionUnit])
        # t.set(Tag.PixelSize, ccfg.PixelSize)

        # DNG Core
        dcfg = self.cfg.dng
        t.set(Tag.DNGVersion, dcfg.DNGVersion)
        t.set(Tag.DNGBackwardVersion, dcfg.DNGBackwardVersion)
        # 16-bit black and white levels
        t.set(Tag.BlackLevel, dcfg.BlackLevel)
        t.set(Tag.WhiteLevel, dcfg.WhiteLevel)

        # Color
        t.set(Tag.ColorMatrix1, self.ccm_rational)
        t.set(Tag.ColorMatrix2, self.ccm_rational)

        t.set(Tag.ForwardMatrix1, self.fm_rational)
        t.set(Tag.ForwardMatrix2, self.fm_rational)

        t.set(Tag.AsShotNeutral, self.as_shot_neutral)

        t.set(Tag.CalibrationIlluminant1, dcfg.CalibrationIlluminant1)
        t.set(Tag.PreviewColorSpace, dcfg.PreviewColorSpace)
        t.set(Tag.BaselineExposure, [dcfg.BaselineExposure])

        # Exif
        t.set(Tag.DateTimeOriginal, self.calculate_dt_from_epoch_gmt(raw_file.stem))
        t.set(Tag.DateTime, self.calculate_dt_from_epoch_gmt(raw_file.stem))
        # DateTimeOriginal
        # DateTime
        return t

    def run(self, tags, raw_image_16: np.ndarray, output_path: Path):
        r = RAW2DNG()
        r.options(tags, path="", compress=False)
        r.convert(raw_image_16, filename=str(output_path))

if __name__ == "__main__":
    # --- Load the raw 16-bit image directly ---
    raw_file = Path("data/raw/MD_1759501672.RAW")

    # --- Load your custom color matrix (3x3) ---
    matrix_path = Path("calibration_results/data/calibration_matrix_20251125_115638.npy")

    # --- Create DNG Tags ---
    config_path = Path("/home/mkutuga/svs-raw-api/experimental/svs_tags.yaml")

    # --- Output path ---
    output_path = Path("data/processed/MD_1759501672")

    # --- LOAD RAW IMAGE ---
    raw_image = np.fromfile(raw_file, dtype=np.uint16).astype(np.uint16)
    raw_image_16 = np.reshape(raw_image, (HEIGHT, WIDTH))
            
    # --- WRITE DNG ---
    svs_dng = SVSRaw2DNG(config_path, matrix_path)
    t = svs_dng.define_tags(raw_file)
    svs_dng.run(t, raw_image_16, output_path)
