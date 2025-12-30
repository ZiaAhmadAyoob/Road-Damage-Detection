import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Road Damage AI",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
st.title("🚧 Road Damage Detection System")
st.markdown("""
    <div style='background-color: #262730; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
        <p style='color: white; margin: 0;'>
            <b>Powered by YOLOv11:</b> Detect potholes, cracks, and road defects with high precision.
        </p>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar Settings ---
st.sidebar.title("⚙️ Control Panel")
st.sidebar.subheader("Model Settings")

# Confidence Slider
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.25, 
    step=0.01,
    help="Adjust how strict the model is when detecting damage."
)

# App Mode Selector
app_mode = st.sidebar.selectbox(
    "Select Mode",
    ["Image Detection", "Video Detection", "Live Webcam"]
)

st.sidebar.markdown("---")
st.sidebar.info("Ensure 'best.pt' is in the root directory.")

# --- Model Loading ---
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model_path = "best.pt"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error("⚠️ Model file 'best.pt' not found. Please upload it to the directory.")
    model = None

# --- Main Logic ---

if model:
    # ---------------- IMAGE MODE ----------------
    if app_mode == "Image Detection":
        st.subheader("📸 Image Analysis")
        uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)

            with col2:
                if st.button("Analyze Image", type="primary"):
                    with st.spinner("Analyzing road surface..."):
                        results = model(img_array, conf=conf_threshold)
                        annotated_img = results[0].plot()
                        
                        st.image(annotated_img, caption="Detected Damage", use_column_width=True)
                        st.success(f"Analysis Complete. Found {len(results[0].boxes)} defects.")

    # ---------------- VIDEO MODE ----------------
    elif app_mode == "Video Detection":
        st.subheader("🎥 Video Analysis")
        video_file = st.file_uploader("Upload a road video", type=["mp4", "avi", "mov"])

        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            
            st.sidebar.markdown("---")
            if st.sidebar.button("Stop Processing"):
                st.session_state.stop = True

            stframe = st.empty()
            cap = cv2.VideoCapture(tfile.name)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=conf_threshold)
                annotated = results[0].plot()
                stframe.image(annotated, channels="BGR", use_column_width=True)

            cap.release()
            st.success("Video processing finished.")

    # ---------------- LIVE WEBCAM MODE ----------------
    elif app_mode == "Live Webcam":
        st.subheader("🔴 Live Road Inspection")
        st.write("Turn on the webcam to detect road damage in real-time.")

        # Helper to use the camera
        run = st.checkbox('Start Camera', value=False)
        
        # Placeholder for the video feed
        frame_window = st.image([])
        
        cap = cv2.VideoCapture(0) # 0 is usually the default webcam

        if run:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture image from camera.")
                    break
                
                # YOLO Prediction
                results = model(frame, conf=conf_threshold)
                
                # Plot results
                annotated_frame = results[0].plot()

                # Display in Streamlit (Convert BGR to RGB for correct colors if needed, 
                # but results[0].plot() usually handles colors well. 
                # We use channels="BGR" to match OpenCV format)
                frame_window.image(annotated_frame, channels="BGR")
        else:
            cap.release()
            st.write("Camera is stopped.")