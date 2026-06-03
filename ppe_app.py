import streamlit as st
from ultralytics import YOLO
import torch
import tempfile
import os
import shutil
from pathlib import Path
import cv2
import imageio_ffmpeg as ffmpeg
import subprocess

# Set Streamlit page configuration
st.set_page_config(
    page_title="PPE Detection & Tracking System",
    page_icon="👷",
    layout="wide"
)


st.title("👷 PPE Detection & Tracking System")

st.markdown("""
Upload an image or video to perform Personal Protective Equipment (PPE)
detection or worker tracking using a custom YOLOv8 model.
""")

# load model function (without caching to avoid issues with video processing)
def load_model():                      # i had to remove cache to separate video and image models
    return YOLO("best.pt")

device = 0 if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    st.write("GPU:", torch.cuda.get_device_name(0))

# sidebar settings
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.radio(
    "Select Mode",
    ["Detection", "Tracking"]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.35,
    step=0.05
)

st.sidebar.markdown("---")

st.sidebar.info(
    f"""
Device: {"GPU 🚀" if torch.cuda.is_available() else "CPU"}

Model: best.pt

Tracker: my_tracker.yaml
"""
)

# file uploader
allowed_types = [
    "jpg",
    "jpeg",
    "png",
    "mp4",
    "avi",
    "mov",
    "mkv"
]

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=allowed_types
)

# Process the uploaded file
if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("🚀 Run Analysis", use_container_width=True):

        model = load_model()

        with st.spinner("Processing... Please wait."):

            # Save uploaded file temporarily
            suffix = Path(uploaded_file.name).suffix

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(uploaded_file.read())
                temp_input_path = temp_file.name

            # Clean previous prediction folders
            runs_dir = Path("runs")

            if runs_dir.exists():
                shutil.rmtree(runs_dir)

            
            # image processing
            if suffix.lower() in [".jpg", ".jpeg", ".png"]:

                model.predict(
                    source=temp_input_path,
                    conf=confidence,
                    device=device,
                    save=True
                )

                output_dir = Path("runs/detect/predict")

                output_files = list(output_dir.glob("*"))

                if output_files:

                    output_image = str(output_files[0])

                    st.subheader("Detection Result")

                    st.image(
                        output_image,
                        use_container_width=True
                    )

                    with open(output_image, "rb") as file:

                        st.download_button(
                            "⬇️ Download Result",
                            file,
                            file_name=os.path.basename(output_image)
                        )

            # video processing
            else:

                st.write("Mode selected:", mode)

                if mode == "Detection":

                    model.predict(
                        source=temp_input_path,
                        conf=confidence,
                        device=device,
                        save=True
                    )

                else:
                    st.write("Confidence:", confidence)
    
                    results= model.track(
                        source=temp_input_path,
                        conf=confidence,
                        device=device,
                        tracker="my_tracker.yaml",
                        imgsz=512,
                        save=True,
                        stream=True
                    )

                    for _ in results:
                        pass

                detect_dir = Path("runs/detect")

                output_videos = []

                for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
                    output_videos.extend(detect_dir.rglob(ext))

                if output_videos:

                    # newest file
                    output_video = max(
                        output_videos,
                        key=lambda p: p.stat().st_mtime
                    )

                    st.success("Video processed successfully!")

                    # Convert AVI to MP4 for browser playback
                    mp4_video = output_video.with_suffix(".mp4")

                    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()

                    subprocess.run(
                        [
                            ffmpeg_exe,
                            "-y",
                            "-i", str(output_video),
                            "-c:v", "libx264",
                            "-preset", "fast",
                            "-crf", "28",
                            str(mp4_video)
                        ],
                        check=True
                    )
                    
                    st.subheader("Processed Video")

                    try:
                        st.video(str(mp4_video))

                    except Exception as e:
                        st.error(f"Could not display video: {e}")

                    with open(mp4_video, "rb") as file:

                        st.download_button(
                            "⬇️ Download Video",
                            file,
                            file_name=mp4_video.name
                        )
                else:

                    st.error("No output video found.")


        st.success("Analysis Complete ✅")
