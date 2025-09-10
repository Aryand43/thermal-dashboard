import numpy as np
import cv2
import time

def detect_thermal_zone(image: np.ndarray) -> tuple[int, int, int, int]:
    start = time.time()
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Input must be a non-empty NumPy array.")
    if len(image.shape) == 2:
        gray = image.copy()
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("Unsupported image format: 2-channel image is not supported.")
    else:
        raise ValueError("Unsupported image format: unexpected number of dimensions.")

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("No thermal zone detected.")

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    end = time.time()
    print(f"Thermal zone detection latency: {end - start:.6f} seconds")
    return (x, y, w, h)

def crop_to_zone(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]
