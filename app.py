import streamlit as st
import os
import cv2
import numpy as np
import tempfile
from preprocessing import detect_and_crop_thermal_zone

st.set_page_config(page_title="Thermal ROI Extractor", layout="centered")
st.title("Thermal ROI Extractor")

OUTPUT_DIR = "ROI"
os.makedirs(OUTPUT_DIR, exist_ok=True)

uploaded_file = st.file_uploader("Upload a thermal video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    progress_bar = st.progress(0)
    frame_display = st.empty()
    saved_count = 0

    for i in range(total_frames):
        success, frame = cap.read()
        if not success:
            continue

        try:
            roi = detect_and_crop_thermal_zone(frame)
            save_path = os.path.join(OUTPUT_DIR, f"roi_{i:05d}.png")
            cv2.imwrite(save_path, roi)
            saved_count += 1
            roi_preview = cv2.resize(roi, (256, 256))
            frame_display.image(roi_preview, caption=f"Frame {i}", channels="BGR")
        except Exception:
            pass

        progress_bar.progress((i + 1) / total_frames)

    cap.release()
    st.write(f"Processed {saved_count} ROI frames. Saved to `{OUTPUT_DIR}/`.")
