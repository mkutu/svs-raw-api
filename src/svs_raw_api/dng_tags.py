from pathlib import Path
import yaml

from typing import Optional, Tuple, List, Dict, Any

from pydantic import BaseModel, model_validator, Field


class ImageConfig(BaseModel):
    SVCamImageWidth: int
    SVCamImageHeight: int
    BitsPerSample: int = 16
    PhotometricInterpretation: int
    Orientation: int
    SamplesPerPixel: int = 1
    CFARepeatPatternDim: Tuple[int, int]
    CFAPattern: Tuple[int, int, int, int]
    RowsPerStrip: Optional[int | str] = None    # may be templated
    TileWidth: Optional[int | str] = None      # may be templated
    TileLength: Optional[int | str] = None

class CameraConfig(BaseModel):
    Make: str
    Model: str
    SerialNumber: str
    LensModel: str
    FocalLength: int
    FocalLengthIn35mmFormat: int
    FocalLengthIn35mmFilm: int
    FNumber: float
    FocalPlaneXResolution: float
    FocalPlaneYResolution: float
    FocalPlaneResolutionUnit: int
    PixelSize: float

class DNGCoreConfig(BaseModel):
    DNGVersion: List[int]
    DNGBackwardVersion: List[int]
    BlackLevel: int
    WhiteLevel: int
    AsShotNeutral: Optional[List[List[int]]] = None
    BaselineExposure: List[int]
    CalibrationIlluminant1: int
    PreviewColorSpace: int

class ExifConfig(BaseModel):
    TimeZoneOffset: int

class SVCamTagConfig(BaseModel):
    image: ImageConfig
    camera: CameraConfig
    dng: DNGCoreConfig
    exif: ExifConfig

    # catch-all for any future sections/tags you didn’t model
    extra: Dict[str, Any] = Field(default_factory=dict)

class ColorConfig(BaseModel):
    calibration_matrix_path: Path
    matrix_den: int = 10000

    # these fields are *computed* after load
    color_matrix: Optional[List[List[int]]] = None
    forward_matrix: Optional[List[List[int]]] = None
    wb_gains: Optional[List[float]] = None
    color_matrix_rational: Optional[List[List[int]]] = None
    forward_matrix_rational: Optional[List[List[int]]] = None
    as_shot_neutral: Optional[List[List[int]]] = None

    @model_validator(mode="after")
    def load_matrices(self) -> "ColorConfig":
        import numpy as np

        data = np.load(self.calibration_matrix_path, allow_pickle=True).item()
        self.color_matrix = data["color_matrix"]
        self.forward_matrix = data["forward_matrix"]
        self.wb_gains = [float(v) for v in data["wb_gains"]]

        transposed_cm = self.color_matrix.T
        self.color_matrix_rational = [
            [int(round(v * self.matrix_den)), self.matrix_den]
            for v in transposed_cm.reshape(-1)
        ]
        transposed_fm = self.forward_matrix.T
        self.forward_matrix_rational = [
            [int(round(v * self.matrix_den)), self.matrix_den]
            for v in transposed_fm.reshape(-1)
        ]

        r, g, b = self.wb_gains
        self.as_shot_neutral = [
            [int(round(self.matrix_den / r)), self.matrix_den],
            [int(round(self.matrix_den / g)), self.matrix_den],
            [int(round(self.matrix_den / b)), self.matrix_den],
        ]
        return self

    @property
    def color_matrix_dng(self) -> List[List[int]]:
        return self.color_matrix_rational

    @property
    def forward_matrix_dng(self) -> List[List[int]]:
        return self.forward_matrix_rational

    @property
    def as_shot_neutral_dng(self) -> List[List[int]]:
        return self.as_shot_neutral


if __name__ == "__main__":

    def load_svcam_config(path: Path) -> SVCamTagConfig:
        raw = yaml.safe_load(path.read_text())
        return SVCamTagConfig(**raw)
    
    config_path = Path("/home/mkutuga/svs-raw-api/experimental/svs_tags.yaml")
    config = load_svcam_config(config_path)
   

    matrix_path = str("calibration_results/data/calibration_matrix_20251125_115638.npy")
    color_config = ColorConfig(calibration_matrix_path=matrix_path)
    config.dng.AsShotNeutral = color_config.as_shot_neutral_dng
    print(config.dng.AsShotNeutral)