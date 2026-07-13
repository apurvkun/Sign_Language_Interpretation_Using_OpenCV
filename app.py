"""
ASL Sign Language Detector - Streamlit App
Author: Apurv
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image

try:
    from utils import ASLDetector
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

st.set_page_config(
    page_title="ASL Sign Language Detector",
    page_icon="✋",
    layout="wide"
)

st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 10px 0;
}
.confidence-bar {
    background-color: #1f77b4;
    height: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    """Load the ASL detector once and cache it across reruns/sessions."""
    return ASLDetector()


def main():
    st.markdown('<h1 class="main-header">✋ ASL Sign Language Detector</h1>', unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Home", "Live Camera Detection", "Image Upload", "Model Info"]
    )

    detector = load_detector()

    if app_mode == "Home":
        show_home()
    elif app_mode == "Live Camera Detection":
        show_live_detection(detector)
    elif app_mode == "Image Upload":
        show_image_upload(detector)
    elif app_mode == "Model Info":
        show_model_info(detector)


def show_home():
    st.markdown("""
    ## Welcome to the ASL Sign Language Detector! 🎉

    This app recognizes American Sign Language (ASL) letters a-z and numbers 0-9
    from your camera or an uploaded image.

    ### How to use:
    1. **Live Camera Detection**: Take a snapshot with your webcam to detect a sign
    2. **Image Upload**: Upload an image containing an ASL sign
    3. **Model Info**: Check what signs the model can recognize

    ### Features:
    - ✅ Browser-based camera capture (works locally *and* when deployed)
    - ✅ ASL alphabet (a-z) recognition
    - ✅ ASL numbers (0-9) recognition
    - ✅ Confidence scoring
    - ✅ Free and open source
    """)

    st.subheader("Supported Signs")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Alphabet (a-z)**")
        st.write("a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z")
    with col2:
        st.write("**Numbers (0-9)**")
        st.write("0, 1, 2, 3, 4, 5, 6, 7, 8, 9")


def _run_prediction_and_display(detector, image_bgr, display_col):
    """Shared helper: run prediction on a BGR numpy image and render results."""
    prediction, confidence, hand_landmarks = detector.predict(image_bgr)

    if hand_landmarks:
        annotated = detector.draw_landmarks(image_bgr, hand_landmarks)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        display_col.image(annotated_rgb, use_container_width=True, caption="Detected hand landmarks")

    if prediction not in ("No hand detected", "Model not loaded", "Prediction error"):
        st.markdown(f"""
        <div class="prediction-box">
            <h2>Prediction: <span style="color: #1f77b4; font-size: 2em;">{prediction}</span></h2>
            <p style="font-size: 1.2em;">Confidence: {confidence:.2%}</p>
            <div class="confidence-bar" style="width: {confidence * 100}%"></div>
        </div>
        """, unsafe_allow_html=True)
    elif prediction == "No hand detected":
        st.warning("❌ No hand detected. Try better lighting and center your hand in frame.")
    elif prediction == "Model not loaded":
        st.error("❌ Model not found! Please train the model first (see Model Info tab).")
    else:
        st.error(f"❌ {prediction}")


def show_live_detection(detector):
    st.header("Live Camera Detection")

    if detector.model is None:
        st.error("❌ Model not found! Please train the model first.")
        st.info("""
        To train the model:
        1. Run: `python train_model.py`
        2. Make sure your ASL dataset is in the `asl_dataset/` folder
        3. The model will be saved as `model/asl_model.pkl`
        """)
        return

    st.info(
        "📷 This uses your browser's camera (works both locally and on Streamlit Cloud). "
        "Show one hand clearly, then click the camera button to capture a snapshot."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        camera_image = st.camera_input("Take a snapshot of your ASL sign")

    with col2:
        if camera_image is not None:
            image = Image.open(camera_image)
            image_np = np.array(image.convert("RGB"))
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            _run_prediction_and_display(detector, image_bgr, col2)
        else:
            st.info("👆 Take a photo to see the prediction here")


def show_image_upload(detector):
    st.header("Upload ASL Image")

    if detector.model is None:
        st.error("❌ Model not found! Please train the model first.")
        return

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image.convert("RGB"))
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Analysis")
            _run_prediction_and_display(detector, image_bgr, col2)


def show_model_info(detector):
    st.header("Model Information")

    if detector.model is None:
        st.error("❌ Model not found! Please train the model first.")
        st.info("""
        To train the model:
        1. Make sure you have the `asl_dataset` folder with 0-9 and a-z subfolders
        2. Run: `python train_model.py`
        3. Wait for training to complete
        """)
        return

    st.success("✅ Model loaded successfully!")
    st.write(f"**Number of classes:** {len(detector.classes)}")
    st.write(f"**Supported signs:** {', '.join(sorted(detector.classes))}")

    st.subheader("How to Train/Retrain the Model")
    st.info(
        "Retraining from inside a deployed Streamlit Cloud app is not reliable "
        "(no dataset access, limited compute/time, and no persistent storage). "
        "Train locally and commit the resulting `model/asl_model.pkl` to your repo instead:\n\n"
        "```\npython train_model.py\n```"
    )


if __name__ == "__main__":
    main()
