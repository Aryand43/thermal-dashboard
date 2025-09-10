import os
import torch
import numpy as np
import cv2
from model import TempPredictorCNN
from utils import load_data_pair  # or just image loader
from tqdm import tqdm

# ---- Config ----
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "temp_predictor_ss316l.pt"
EVAL_DIR = "eval_data/"  # Folder with unseen ROI frames
IMAGE_SIZE = (128, 128)
PX_PER_SEC = 133
PX_SIZE_UM = 31.34

# ---- Load model ----
model = TempPredictorCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---- Predict on eval data ----
image_files = sorted([f for f in os.listdir(EVAL_DIR) if f.endswith(('.png', '.jpg', '.tif'))])
predictions = []

with torch.no_grad():
    for file in tqdm(image_files):
        img_path = os.path.join(EVAL_DIR, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMAGE_SIZE).astype(np.float32) / 255.0
        img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1, 1, H, W]
        pred = model(img_tensor).squeeze().cpu().numpy()  # [H, W]
        predictions.append(pred)

# ---- Compute Thermal Velocity ----
def compute_thermal_velocity(temp_stack, px_per_sec=133, px_size_um=31.34):
    """
    temp_stack: [T x H x W] predicted temperature over time
    """
    velocities = []
    for t in range(1, len(temp_stack)):
        delta_temp = temp_stack[t] - temp_stack[t-1]
        temp_gradient = np.abs(delta_temp).mean()
        velocity = temp_gradient * px_per_sec * px_size_um  # µm/s
        velocities.append(velocity)
    return np.mean(velocities), velocities

temp_stack = np.stack(predictions, axis=0)
mean_velocity, velocity_series = compute_thermal_velocity(temp_stack)

print(f"\nMean Thermal Velocity: {mean_velocity:.2f} µm/s")
