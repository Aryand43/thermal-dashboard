import os
import cv2
import numpy as np
import pandas as pd
import torch
def upscale_label_2x(label: np.ndarray) -> np.ndarray:
    """Repeat every pixel in the label 2×2 to match image dimensions."""
    label = np.repeat(label, repeats=2, axis=0)
    label = np.repeat(label, repeats=2, axis=1)
    return label

def load_dataset(roi_dir: str, label_dir: str):
    print("ROI folder:", roi_dir)
    print("Label folder:", label_dir)

    roi_files = sorted([f for f in os.listdir(roi_dir) if f.endswith('.tiff') or f.endswith('.tif')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.xlsx') or f.endswith('.csv')])

    print("Found ROI files:", roi_files)
    print("Found Label files:", label_files)

    dataset = []

    roi_files = sorted([f for f in os.listdir(roi_dir) if f.endswith(".tiff") or f.endswith(".tif")])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".xlsx") or f.endswith(".csv")])

    for roi_file, label_file in zip(roi_files, label_files):
        # Load and normalize image
        img_path = os.path.join(roi_dir, roi_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        # Load label
        label_path = os.path.join(label_dir, label_file)
        df = pd.read_excel(label_path, header=None)
        label = df.to_numpy().astype(np.float32)

        # Check shape
        h, w = img.shape
        assert label.shape[0] * 2 == h and label.shape[1] * 2 == w, \
            f"Shape mismatch. Image: {img.shape}, Label: {label.shape}"

        # Upscale label
        label = upscale_label_2x(label)
        label = label / label.max()  # normalize to [0, 1]

        # Convert to PyTorch tensors: shape = [1, H, W]
        img_tensor = torch.tensor(img).unsqueeze(0)       # [1, 764, 576]
        label_tensor = torch.tensor(label).unsqueeze(0)   # [1, 764, 576]

        dataset.append((img_tensor, label_tensor))

    return dataset
