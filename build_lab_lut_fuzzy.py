# build_lab_lut_fuzzy.py (Debug Mode)

import os
import cv2
import numpy as np
import pandas as pd
from skimage import color
from scipy.ndimage import zoom
from tqdm import tqdm

DATA_DIR = "data"
LABELS_DIR = "labels"
OUTPUT_CSV = "lab_lut.csv"
SAMPLE_RATE = 4  # every Nth pixel

# Discover files
data_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.tif', '.tiff'))])
label_files = sorted([f for f in os.listdir(LABELS_DIR) if f.lower().endswith('.csv')])

# Debug checks
print(f"[INFO] Found {len(data_files)} image(s) in '{DATA_DIR}'")
print(f"[INFO] Found {len(label_files)} label file(s) in '{LABELS_DIR}'")
print(f"[DEBUG] First 3 image files: {data_files[:3]}")
print(f"[DEBUG] First 3 label files: {label_files[:3]}")

if len(data_files) != len(label_files):
    print(f"[WARN] Mismatch in number of files — only processing up to min length: {min(len(data_files), len(label_files))}")

entries = []

# Main pairing loop
for img_file, label_file in tqdm(zip(data_files, label_files), total=min(len(data_files), len(label_files))):
    img_path = os.path.join(DATA_DIR, img_file)
    label_path = os.path.join(LABELS_DIR, label_file)

    print(f"\n[PAIR] {img_file} <--> {label_file}")

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[ERROR] Couldn't load image: {img_path}")
        continue

    h, w, _ = img.shape
    print(f"[INFO] Image shape: {h}x{w}")
    img_lab = color.rgb2lab(img / 255.0)

    try:
        label_csv = pd.read_csv(label_path, header=None)
        label_array = label_csv.values.astype(np.float32)
    except Exception as e:
        print(f"[ERROR] Failed reading CSV: {label_path} — {e}")
        continue

    print(f"[INFO] Label shape: {label_array.shape}")

    # Resize label to match image
    zoom_y = h / label_array.shape[0]
    zoom_x = w / label_array.shape[1]
    label_resized = zoom(label_array, (zoom_y, zoom_x), order=1)

    # Sample and store entries
    added = 0
    for y in range(0, h, SAMPLE_RATE):
        for x in range(0, w, SAMPLE_RATE):
            L, A, B = img_lab[y, x]
            temp = label_resized[y, x]
            if np.isnan(temp):
                continue
            entries.append([L, A, B, temp])
            added += 1

    print(f"[INFO] Added {added} points from this pair.")

# Save output
df = pd.DataFrame(entries, columns=["L", "A", "B", "Temp"])
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDONE. Saved LUT with {len(df)} entries to '{OUTPUT_CSV}'")
