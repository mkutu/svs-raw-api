import numpy as np

data = "calibration_results/data/calibration_data_20251119_171642.npy"
matrix = "calibration_results/data/calibration_matrix_20251119_171642.npy"

# open both files and inspect their contents
cal_data = np.load(data, allow_pickle=True).item()
cal_matrix = np.load(matrix, allow_pickle=True)
print("Calibration Data Keys:", cal_data.keys())
print("Calibration Matrix Shape:", cal_matrix.shape)
print("Calibration Matrix:", cal_matrix)

# Inspect specific entries in cal_data
measured_colors = cal_data.get('measured_colors')
print("Measured Colors Shape:", measured_colors.shape)
print("Measured Colors:", measured_colors)

corrected_colors = cal_data.get('corrected_colors')
print("Corrected Colors Shape:", corrected_colors.shape)
print("Corrected Colors:", corrected_colors)

# Multiple each number in calibration matrix by 10000 and print
scaled_matrix = cal_matrix * 10000
print("Scaled Calibration Matrix:", scaled_matrix)