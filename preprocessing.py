import os
import cv2
import numpy as np
import time

DATA_DIR = "data"
OUTPUT_DIR = "ROI"

def detect_and_crop_thermal_zone(image: np.ndarray) -> np.ndarray:
    start = time.time()

    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Input must be a non-empty NumPy array.")

    # Convert to grayscale
    if len(image.shape) == 2:
        gray = image.copy()
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("Unsupported image format.")
    else:
        raise ValueError("Unsupported image format.")

    # Use very strict high threshold (top 2 pixel intensities only)
    high_thresh_value = np.percentile(gray, 99.9)
    _, hot_thresh = cv2.threshold(gray, high_thresh_value, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(hot_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No thermal zone detected.")

    # Filter out small contours (e.g. noise, dead pixels)
    contours = [c for c in contours if cv2.contourArea(c) > 5]

    if not contours:
        raise RuntimeError("No valid melt pool contour found.")

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Optional tight padding (or remove if not needed)
    pad = 4
    x = max(x - pad, 0)
    y = max(y - pad, 0)
    w = min(w + 2 * pad, image.shape[1] - x)
    h = min(h + 2 * pad, image.shape[0] - y)

    cropped = image[y:y+h, x:x+w]
    print(f"[ROI] max={np.max(gray)}, thresh={high_thresh_value:.2f}, BBox=(x={x}, y={y}, w={w}, h={h}), Time={time.time() - start:.3f}s")
    return cropped

def is_image_file(filename):
    return filename.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"))

def process_all_images():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_files = [f for f in os.listdir(DATA_DIR) if is_image_file(f)]
    print(f"Found {len(image_files)} image(s) in {DATA_DIR}/")

    for i, filename in enumerate(image_files):
        input_path = os.path.join(DATA_DIR, filename)
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"[SKIP] Failed to load {filename}")
            continue

        try:
            cropped = detect_and_crop_thermal_zone(image)
        except Exception as e:
            print(f"[ERROR] Skipping {filename} due to: {e}")
            continue

        base_filename = os.path.splitext(filename)[0]
        save_path = os.path.join(OUTPUT_DIR, f"{base_filename}_roi.tiff")
        cv2.imwrite(save_path, cropped)
        print(f"[SAVED] {save_path}")

def guided_filter_and_unsharp_mask(image: np.ndarray) -> np.ndarray:
    import time
    start = time.time()

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    guide = image.copy()
    filtered = cv2.ximgproc.guidedFilter(guide=guide, src=image, radius=6, eps=40, dDepth=-1)

    sharp = cv2.addWeighted(image, 1.8, filtered, -0.8, 0)
    print(f"Guided sharpening latency: {time.time() - start:.6f} seconds")
    return np.clip(sharp, 0, 255).astype(np.uint8)

def apply_edsr_and_lab_refinement(image: np.ndarray, model_path: str = 'EDSR_x4.pb', scale: int = 2) -> np.ndarray:
    import time
    start = time.time()

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel('edsr', scale)
    upscaled = sr.upsample(image)

    lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(cv2.GaussianBlur(l, (3, 3), 0))
    sharpened = cv2.Laplacian(l, cv2.CV_64F)
    l_final = cv2.addWeighted(l, 1.2, sharpened.astype(np.uint8), -0.2, 0)
    final = cv2.merge((l_final, a, b))
    upscaled_image = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)

    print(f"Super resolution latency: {time.time() - start:.6f} seconds")
    return upscaled_image

if __name__ == "__main__":
    process_all_images()
