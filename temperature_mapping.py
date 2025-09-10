import cv2
import numpy as np
import pandas as pd
from skimage import color
from scipy.spatial import KDTree

def load_lab_lut(csv_path="lab_lut.csv"):
    """
    Load LAB color-temperature LUT from CSV.
    Returns KDTree-ready LAB array and corresponding temperatures.
    """
    df = pd.read_csv(csv_path)
    lab = df[["L", "A", "B"]].values
    temps = df["Temp"].values
    return lab, temps

def generate_temp_map(image_bgr, lab_colors, temps, delta_e_thresh=8.0, downscale_factor=4):
    """
    Downsample before mapping, then upsample back.
    ~10x faster for live processing.
    """
    img_rgb = image_bgr[..., ::-1] / 255.0
    img_lab = color.rgb2lab(img_rgb)

    h, w, _ = img_lab.shape
    h_d, w_d = h // downscale_factor, w // downscale_factor
    small_lab = cv2.resize(img_lab, (w_d, h_d), interpolation=cv2.INTER_AREA)
    flat_lab = small_lab.reshape(-1, 3)

    tree = KDTree(lab_colors)
    dist, idx = tree.query(flat_lab)

    temp_map_small = np.full((h_d * w_d), np.nan)
    matched = dist <= delta_e_thresh
    temp_map_small[matched] = temps[idx[matched]]
    temp_map_small = temp_map_small.reshape(h_d, w_d)

    # Upsample to original size
    temp_map = cv2.resize(temp_map_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return temp_map