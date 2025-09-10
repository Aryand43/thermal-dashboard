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
from temperature_mapping import load_lab_lut, generate_temp_map
import matplotlib.pyplot as plt

st.set_page_config(page_title="Melt Pool Thermal ROI Extractor", layout="centered")
st.title("Melt Pool Thermal Dashboard")
st.subheader("Upload a single-track thermal video or image")

@st.cache_resource
def load_lut_once():
    return load_lab_lut("lab_lut.csv")

lab_colors, temps = load_lut_once()

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
                    try:
                        temp_map = generate_temp_map(roi, lab_colors, temps)
                        np.save(os.path.join(output_dir, f"temp_map_{count:05d}.npy"), temp_map)
                    except Exception as e:
                        st.warning(f"[TempMap Error: Frame {count}] {e}")

                    from matplotlib import pyplot as plt
                    import matplotlib
                    matplotlib.use("Agg")

                    import io
                    from PIL import Image

                    try:
                        # Generate temperature map from LAB
                        temp_map = generate_temp_map(roi, lab_colors, temps)
                        temp_map_display = cv2.resize(temp_map, (roi.shape[1], roi.shape[0]))

                        # Normalize and apply heatmap color
                        norm_temp = cv2.normalize(temp_map_display, None, 0, 255, cv2.NORM_MINMAX)
                        heatmap = cv2.applyColorMap(norm_temp.astype(np.uint8), cv2.COLORMAP_INFERNO)

                        # Convert both to RGB for display
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                        # Plot side-by-side vertically
                        fig, ax = plt.subplots(2, 1, figsize=(4, 6))
                        ax[0].imshow(roi_rgb)
                        ax[0].set_title(f"Frame {count}: ROI")
                        ax[0].axis("off")

                        ax[1].imshow(heatmap_rgb)
                        ax[1].set_title("Estimated Temperature Map")
                        ax[1].axis("off")

                        buf = io.BytesIO()
                        plt.tight_layout()
                        plt.savefig(buf, format="png")
                        buf.seek(0)
                        image = Image.open(buf)
                        frame_display.image(image, use_column_width=True)
                        plt.close(fig)

                    except Exception as e:
                        print(f"[TempMap Error: Frame {count}] {e}")
                        continue
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

            try:
                temp_map = generate_temp_map(roi, lab_colors, temps)
                np.save(os.path.join(output_dir, "temp_map_image.npy"), temp_map)
            except Exception as e:
                st.warning(f"[TempMap Error] {e}")

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

        st.subheader("Sample Temperature Maps")

        for img_file in output_images[:3]:
            temp_path = os.path.join(output_dir, img_file.replace("roi_", "temp_map_").replace(".png", ".npy"))
            if os.path.exists(temp_path):
                temp = np.load(temp_path)

                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(temp, cmap="plasma")
                step = max(1, temp.shape[0] // 10)
                for y in range(0, temp.shape[0], step):
                    for x in range(0, temp.shape[1], step):
                        val = temp[y, x]
                        if np.isfinite(val):
                            ax.text(x, y, f"{val:.0f}", ha='center', va='center', color='white', fontsize=6)

                fig.colorbar(im, ax=ax, label="Temperature (°C)")
                st.pyplot(fig)


                # Temperature stats
                finite_temp = temp[np.isfinite(temp)]
                if finite_temp.size > 0:
                    st.write(f"**{img_file} Stats:**")
                    st.write(f"- Min Temp: {np.nanmin(temp):.2f} °C")
                    st.write(f"- Max Temp: {np.nanmax(temp):.2f} °C")
                    st.write(f"- Mean Temp: {np.nanmean(temp):.2f} °C")
                else:
                    st.write(f"**{img_file}** has no valid temperature values.")

                # Optional: Display raw °C matrix as table
                with st.expander(f"Show temperature matrix for {img_file}"):
                    st.dataframe(np.round(temp, 2), use_container_width=True)

                # Optional: Download temperature map
                st.download_button(
                    label="Download Temp Matrix (.npy)",
                    data=temp.tobytes(),
                    file_name=img_file.replace("roi_", "temp_map_").replace(".png", ".npy"),
                    mime="application/octet-stream"
                )

