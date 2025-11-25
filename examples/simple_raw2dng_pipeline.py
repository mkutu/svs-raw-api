from pathlib import Path
import numpy as np

from svs_raw_api import SVSRaw2DNG, HEIGHT, WIDTH


# --- Load the raw 16-bit image directly ---
raw_file = Path("data/raw/MD_1759501672.RAW")

# --- Load your custom color matrix (3x3) ---
matrix_path = Path("data/processed/MD_calibration_matrix_optimized.npy")

# --- Create DNG Tags ---
config_path = Path("examples/svs_tags.yaml")

# --- Output path ---
output_path = Path("data/processed/MD_1759501672_final")

# --- LOAD RAW IMAGE ---
raw_image = np.fromfile(raw_file, dtype=np.uint16).astype(np.uint16)
raw_image_16 = np.reshape(raw_image, (HEIGHT, WIDTH))
        
# --- WRITE DNG ---
svs_dng = SVSRaw2DNG(config_path, matrix_path)
t = svs_dng.define_tags(raw_file)
svs_dng.run(t, raw_image_16, output_path)