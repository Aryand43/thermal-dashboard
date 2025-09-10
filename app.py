import streamlit as st
import os
import cv2
import numpy as np
import tempfile
from datetime import datetime
from preprocessing import (
    detect_and_crop_thermal_zone,
    guided_filter_and_unsharp_mask,
    apply_edsr_and_lab_refinement
)

st.set_page_config(page_title="Melt Pool Thermal ROI Extractor", layout="centered")
st.title("Melt Pool Thermal Dashboard")
st.subheader("Upload a single-track thermal video or image")

uploaded_file = st.file_uploader("Upload Thermal Image/Video", type=["mp4", "avi", "mov", "mpeg", "mkv", "jpg", "jpeg", "png", "tif", "tiff"])

enhancement_option = st.selectbox(
    "Choose enhancement mode:",
    ("None", "Guided Filter + Unsharp Mask", "EDSR + LAB Refinement")
)

model_path = "EDSR_x4.pb"
if enhancement_option == "EDSR + LAB Refinement" and not os.path.exists(model_path):
    st.error("Missing EDSR model file: EDSR_x4.pb")
    st.stop()

if uploaded_file:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    file_bytes = uploaded_file.read()
    file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(file_path, "wb") as out_file:
        out_file.write(file_bytes)

    if uploaded_file.type.startswith("video"):
        st.video(file_path)
        cap = cv2.VideoCapture(file_path)
        count = 0
        success = True
        progress_bar = st.progress(0)
        frame_display = st.empty()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with st.spinner("Processing video frames..."):
            while success:
                success, frame = cap.read()
                if not success:
                    break
                try:
                    roi = detect_and_crop_thermal_zone(frame)

                    if enhancement_option == "Guided Filter + Unsharp Mask":
                        roi = guided_filter_and_unsharp_mask(roi)
                    elif enhancement_option == "EDSR + LAB Refinement":
                        roi = apply_edsr_and_lab_refinement(roi, model_path=model_path)

                    save_path = os.path.join(output_dir, f"roi_{count:05d}.png")
                    cv2.imwrite(save_path, roi)
                    roi_preview = cv2.resize(roi, (256, 256))
                    frame_display.image(roi_preview, caption=f"Frame {count}", channels="BGR")
                    count += 1
                except:
                    continue
                progress_bar.progress((count + 1) / total_frames)

        cap.release()
        st.success(f"Processed and saved {count} ROI frame(s) to '{output_dir}'")

    else:
        image_np = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        try:
            roi = detect_and_crop_thermal_zone(image_np)

            if enhancement_option == "Guided Filter + Unsharp Mask":
                roi = guided_filter_and_unsharp_mask(roi)
            elif enhancement_option == "EDSR + LAB Refinement":
                roi = apply_edsr_and_lab_refinement(roi, model_path=model_path)

            save_path = os.path.join(output_dir, "roi_image.png")
            cv2.imwrite(save_path, roi)
            st.image(roi, caption="Extracted ROI", use_column_width=True)
            st.success(f"Cropped ROI saved to '{output_dir}/roi_image.png'")
        except Exception as e:
            st.error(f"Processing failed: {e}")

    if os.path.exists(output_dir):
        st.subheader("Extracted ROI Output(s)")
        output_images = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
        for img_file in output_images:
            img_path = os.path.join(output_dir, img_file)
            st.image(img_path, caption=img_file, use_column_width=True)