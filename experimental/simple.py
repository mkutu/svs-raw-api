from pathlib import Path

import numpy as np
import cv2

from svs_raw_api import PATCH_NAMES
from svs_raw_api import load_raw_image, demosaic_image, apply_color_correction
from experimental.wb_utils import compute_white_balance_gains, apply_white_balance



raw_colorchecker        = Path('data/raw/MD_1759501672.RAW')
colorchecker_matrix     = Path('data/matrices/calibration_matrix_test.npy')
data                    = Path("data/calibration/calibration_data_test.npy")
output_dir              = Path('data/processed')
output_dir.mkdir(parents=True, exist_ok=True)

ccm                     = np.load(colorchecker_matrix, allow_pickle=True)
cal_data                = np.load(data, allow_pickle=True).item()

patch_colors            = cal_data.get('measured_colors')
grey_patch_names        = PATCH_NAMES[19:24]  # White through Black

checker_top_left        = (5258, 5863)
checker_bottom_right    = (6043, 6817)


# Compute white-balance gains
print("Computing white-balance gains...")
wb_gains = compute_white_balance_gains(
    patch_colors=patch_colors,
    patch_names=PATCH_NAMES,
    grey_patch_names=grey_patch_names,
)

print("Loading raw image...")
bayer_linear = load_raw_image(raw_colorchecker)        # [H, W], linear [0,1]
print("Demosaicing image...")
rgb_linear   = demosaic_image(bayer_linear)    # [H, W, 3], linear [0,1]
print("Applying white balance...")
rgb_wb = apply_white_balance(rgb_linear, wb_gains, clip=True)
print("Applying color correction matrix...")
rgb_ccm = apply_color_correction(rgb_wb, ccm)


# convert from rgb to bgr for cv2.imwrite
bgr_wb = rgb_wb[:, :, ::-1]
# resize for faster processing
bgr_wb_small = cv2.resize(bgr_wb, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
wb_output_path = output_dir / 'MD_1759501672_wb_preview.png'
cv2.imwrite(str(wb_output_path), (bgr_wb_small * 255).astype(np.uint8))
# wb_output_path = output_dir / 'MD_1759501672_wb_fullres.png'
# cv2.imwrite(str(wb_output_path), (bgr_wb * 255).astype(np.uint8))


# convert from rgb to bgr for cv2.imwrite
bgr_ccm = rgb_ccm[:, :, ::-1]
# resize for faster processing
bgr_ccm_small = cv2.resize(bgr_ccm, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
# save output
ccm_output_path = output_dir / 'MD_1759501672_ccm_preview.png'
cv2.imwrite(str(ccm_output_path), (bgr_ccm_small * 255).astype(np.uint8))
# ccm_output_path = output_dir / 'MD_1759501672_ccm_fullres.png'
# cv2.imwrite(str(ccm_output_path), (bgr_ccm * 255).astype(np.uint8))