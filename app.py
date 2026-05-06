import streamlit as st
import numpy as np
from PIL import Image
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lung X-Ray Disease Detector",
    page_icon="",
    layout="centered",
)

# ── Model loading (cached) ────────────────────────────────────────────────────
MODEL_PATH = "lung_disease_model.h5"
HF_REPO    = "lakshyalol/customann1"
HF_FILE    = "lung_disease_model.h5"

@st.cache_resource(show_spinner=False)
def load_model():
    """Download from HuggingFace if not cached, then load."""
    import tensorflow as tf
    from huggingface_hub import hf_hub_download

    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model from HuggingFace (~153 MB)…"):
            path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE,
                                   local_dir=".")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# ── Class labels ──────────────────────────────────────────────────────────────
# Adjust this list to match your model's training classes
CLASS_NAMES = ["Normal", "Pneumonia"]   # ← update if you have more classes

IMG_SIZE = (224, 224)   # ← update if your model uses a different input size

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    """Resize, convert to RGB, normalise [0,1], add batch dim."""
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, H, W, 3)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🫁 Lung X-Ray Disease Detector")
st.caption("Custom ANN · IIT Bhilai Numerical Methods Project")

st.markdown(
    """
    Upload a chest X-ray image and the model will classify it. 
    > **Repo:** [swagat27/Lung-XRay-Numerical-Methods](https://github.com/swagat27/Lung-XRay-Numerical-Methods)
    """
)

uploaded = st.file_uploader(
    "Upload a chest X-ray (JPG / PNG / JPEG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    with col2:
        with st.spinner("Loading model…"):
            model = load_model()

        with st.spinner("Running inference…"):
            tensor  = preprocess(image)
            preds   = model.predict(tensor, verbose=0)   # shape: (1, n_classes) or (1,1)

        # ── Handle sigmoid (binary) vs softmax (multi-class) output ──────────
        if preds.shape[-1] == 1:
            # Binary sigmoid output
            prob_positive = float(preds[0][0])
            probs = [1 - prob_positive, prob_positive]
        else:
            probs = preds[0].tolist()

        # Pad / trim class list if model has different num outputs
        labels = CLASS_NAMES[:len(probs)] if len(CLASS_NAMES) >= len(probs) \
                 else CLASS_NAMES + [f"Class {i}" for i in range(len(CLASS_NAMES), len(probs))]

        pred_idx   = int(np.argmax(probs))
        pred_label = labels[pred_idx]
        confidence = probs[pred_idx] * 100

        st.subheader("Prediction")
        if pred_label.lower() == "normal":
            st.success(f"✅ **{pred_label}**  ({confidence:.1f}% confidence)")
        else:
            st.error(f"⚠️ **{pred_label}**  ({confidence:.1f}% confidence)")

        st.subheader("Class Probabilities")
        for label, prob in zip(labels, probs):
            st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")

    st.divider()
    st.subheader("🔬 Numerical Methods Applied")
    st.markdown(
        """
        This project uses numerical techniques for X-ray preprocessing:

        | Technique | Application |
        |---|---|
        | **Gaussian Filtering** | Noise reduction via convolution |
        | **Finite Differences** | Edge detection (Sobel / Laplacian) |
        | **FFT-based Filtering** | Frequency-domain enhancement |
        | **Histogram Equalisation** | Contrast normalisation |
        | **LU Decomposition** | Linear system solving in filters |
        """
    )

else:
    st.info("👆 Upload an X-ray image to get started.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        **Lung X-Ray Disease Detector**

        A custom ANN trained on chest X-ray images, combined with
        classical numerical methods for medical image analysis.

        **Classes detected:**
        """
    )
    for c in CLASS_NAMES:
        st.markdown(f"- {c}")

    st.divider()
    st.markdown(
        """
        **Stack**
        - TensorFlow / Keras
        - Streamlit
        - HuggingFace Hub
        - NumPy · Pillow
        """
    )
    st.divider()
    st.caption("IIT Bhilai · Numerical Methods Term Project")
