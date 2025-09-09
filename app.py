import streamlit as st
import cv2
import os
import tempfile
from PIL import Image
import shutil

st.set_page_config(page_title="Video Frame Extractor", layout="centered")
st.title("Video Frame Extractor @ 80 FPS")
st.write("Upload a video to extract frames every 1/80 seconds.")

uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)

    if not cap.isOpened():
        st.error("Failed to open video.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        st.success(f"Video duration: {duration:.2f} sec | FPS: {fps:.2f}")

        frame_time_gap = 1 / 80
        timestamps = [t for t in [i * frame_time_gap for i in range(int(duration * 80))] if t <= duration]

        output_dir = "capture"
        os.makedirs(output_dir, exist_ok=True)

        saved = 0
        progress = st.progress(0)

        for i, t in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            filename = os.path.join(output_dir, f"frame_{saved:05d}.png")
            img.save(filename)
            saved += 1
            progress.progress(i / len(timestamps))

        cap.release()
        st.success(f"Extracted {saved} frames at 80 FPS.")

        st.subheader("Sample Extracted Frames:")
        for i in range(min(saved, 5)):
            img_path = os.path.join(output_dir, f"frame_{i:05d}.png")
            st.image(img_path, caption=f"Frame {i}", use_column_width=True)

        zip_path = os.path.join(output_dir, "extracted_frames.zip")
        shutil.make_archive(zip_path.replace(".zip", ""), 'zip', output_dir)

        with open(zip_path, "rb") as f:
            st.download_button("Download All Frames (.zip)", f, file_name="extracted_frames.zip")
