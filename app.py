import os

import cv2
import numpy as np
import joblib
import streamlit as st

# ── DeepFace import (suppress TF noise) ───────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    from deepface import DeepFace
except ImportError as e:
    st.error(f"❌ DeepFace import failed: {e}")
    st.stop()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition – Deep Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    color: #e0e0e0;
}
h1 { color: #00d4ff; font-weight: 900; letter-spacing: -1px; }
h2, h3 { color: #a0c4ff; font-weight: 700; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(0,212,255,0.15);
}

.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(12px);
}
.result-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(58,123,213,0.08));
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 20px;
    padding: 30px 20px;
    text-align: center;
    box-shadow: 0 0 40px rgba(0,212,255,0.1);
    animation: glow 2s ease-in-out infinite alternate;
}
@keyframes glow {
    from { box-shadow: 0 0 20px rgba(0,212,255,0.1); }
    to   { box-shadow: 0 0 40px rgba(0,212,255,0.3); }
}
.person-name {
    font-size: 2.4rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00d4ff, #00ff88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 12px 0;
}
.conf-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 30px;
    background: linear-gradient(90deg, #00d4ff, #3a7bd5);
    color: white;
    font-weight: 700;
    font-size: 1.1rem;
    margin-top: 8px;
}
.unknown-card {
    background: rgba(255,80,80,0.08);
    border: 1px solid rgba(255,80,80,0.3);
    border-radius: 20px;
    padding: 30px 20px;
    text-align: center;
}
.person-chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    margin: 4px;
    font-size: 0.85rem;
    color: #a0c4ff;
}
.debug-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 12px;
    font-family: monospace;
    font-size: 0.8rem;
    color: #888;
    margin-top: 10px;
}
img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE      = (224, 224)
DATA_DIR      = 'dataset_cleaned'
MODEL_FILE    = 'best_model.pkl'
FACENET_MODEL = 'Facenet'

# ── Load pre-trained model (cached) ──────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the pre-trained SVM model from best_model.pkl."""
    model = joblib.load(MODEL_FILE)
    return model

# ── OpenCV Haar Cascade face detector (cached) ────────────────────────────────
# Uses the cascade bundled inside OpenCV — no extra downloads or system libs.
@st.cache_resource
def get_face_detector(scale_factor, min_neighbors):
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    return detector

# ── Crop face from image ──────────────────────────────────────────────────────
def crop_face(img_bgr, detector, scale_factor=1.1, min_neighbors=5):
    h, w   = img_bgr.shape[:2]
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces  = detector.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return None, None
    # Pick largest face
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    # Add 10% padding
    pad_x = int(fw * 0.10)
    pad_y = int(fh * 0.10)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + fw + pad_x)
    y2 = min(h, y + fh + pad_y)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size > 0:
        return cv2.resize(crop, IMG_SIZE), (x1, y1, x2, y2)
    return None, None

# ── Helper: known people list ─────────────────────────────────────────────────
def get_class_names():
    if os.path.exists(DATA_DIR):
        return sorted([c for c in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR, c))])
    return []

# ── Extract embedding + predict ───────────────────────────────────────────────
def predict_face(face_img_bgr, model, unknown_thresh):
    """Run FaceNet + SVM on a cropped face image. Returns (name, conf, probs, classes)."""
    temp_path = "_temp_face.jpg"
    cv2.imwrite(temp_path, face_img_bgr)
    try:
        objs = DeepFace.represent(
            img_path=temp_path,
            model_name=FACENET_MODEL,
            enforce_detection=False
        )
        if not objs:
            return None, 0.0, None, None

        emb      = np.array(objs[0]['embedding']).reshape(1, -1)
        # L2 normalize — CRITICAL: must match training normalization
        norm     = np.linalg.norm(emb)
        if norm > 0:
            emb_norm = emb / norm
        else:
            emb_norm = emb

        probs    = model.predict_proba(emb_norm)[0]
        classes  = model.classes_
        best_idx = int(np.argmax(probs))
        pred     = classes[best_idx]
        conf     = float(probs[best_idx])

        if conf < unknown_thresh:
            return "unknown", conf, probs, classes
        return pred, conf, probs, classes
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL — fail fast if not found
# ═══════════════════════════════════════════════════════════════════════════════
if not os.path.exists(MODEL_FILE):
    st.error("❌ **`best_model.pkl` not found!**  Run `train_best_model.py` first.")
    st.stop()

ml_model    = load_model()
class_names = get_class_names()

# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<h1>🧠 Deep Learning Face Recognition</h1>", unsafe_allow_html=True)
st.markdown("##### FaceNet embeddings · SVM classifier · OpenCV face detection")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("<h2>⚙️ Settings</h2>", unsafe_allow_html=True)

min_confidence = st.sidebar.slider("Face Detection Threshold", 0.10, 1.00, 0.40, 0.05)
unknown_thresh = st.sidebar.slider(
    "Min Confidence to Accept", 0.10, 1.00, 0.35, 0.05,
    help="Predictions below this are shown as Unknown"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Model Info")

model_type = type(ml_model).__name__
kernel_info = ""
if hasattr(ml_model, 'kernel'):
    kernel_info = f"  \n🔧 Kernel: `{ml_model.kernel}`, C={ml_model.C}"

st.sidebar.success(f"✅ Model: `{model_type}`{kernel_info}")
st.sidebar.info(f"Recognizes **{len(class_names)}** people")

st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Known People")
for name in class_names:
    st.sidebar.markdown(
        f"<span class='person-chip'>👤 {name.replace('_',' ').title()}</span>",
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  STATS ROW
# ═══════════════════════════════════════════════════════════════════════════════
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(f"""<div class='metric-card'>
        <h2 style='color:#00d4ff;font-size:2.2rem;margin:0'>{len(class_names)}</h2>
        <p style='margin:0;color:#888'>People in Dataset</p>
    </div>""", unsafe_allow_html=True)

with col_b:
    total_imgs = sum(
        len([f for f in os.listdir(os.path.join(DATA_DIR, c))
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for c in class_names
        if os.path.isdir(os.path.join(DATA_DIR, c))
    ) if class_names else 0
    st.markdown(f"""<div class='metric-card'>
        <h2 style='color:#00ff88;font-size:2.2rem;margin:0'>{total_imgs}</h2>
        <p style='margin:0;color:#888'>Training Images</p>
    </div>""", unsafe_allow_html=True)

with col_c:
    st.markdown(f"""<div class='metric-card'>
        <h2 style='color:#00d4ff;font-size:1.6rem;margin:0'>✅ Ready</h2>
        <p style='margin:0;color:#888'>FaceNet (128D) + SVM</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Initialize session state
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ═══════════════════════════════════════════════════════════════════════════════
#  INPUT METHOD Selection
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["📁 File Upload", "📷 Live Webcam Capture"])

uploaded_file = None

with tab1:
    st.markdown("### 📸 Upload a Face Image")
    file_upload = st.file_uploader(
        "Drag & drop or click to browse — supports JPG · JPEG · PNG · WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"uploader_{st.session_state.uploader_key}"
    )
    if file_upload is not None:
        uploaded_file = file_upload

with tab2:
    st.markdown("### 📷 Live Webcam Capture")
    webcam_upload = st.camera_input(
        "Capture a face photo from your webcam", 
        key=f"camera_{st.session_state.uploader_key}"
    )
    if webcam_upload is not None:
        uploaded_file = webcam_upload

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr  = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 🖼️ Input Image")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.markdown("#### 🔍 Recognition Result")

        detector = get_face_detector(1.1, int(min_confidence * 10))
        cropped_face, bbox = crop_face(image_bgr, detector)

        if cropped_face is None:
            st.markdown("""<div class='unknown-card'>
                <h2 style='color:#ff6b6b'>❌ No Face Detected</h2>
                <p>Try a clearer image or lower the detection threshold in the sidebar.</p>
            </div>""", unsafe_allow_html=True)
        else:
            # Show the cropped face that gets analyzed
            st.image(
                cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB),
                caption="Detected & Cropped Face (224×224)",
                width=160
            )

            with st.spinner("⏳ Extracting FaceNet embedding & predicting…"):
                try:
                    pred, conf, probs, classes = predict_face(
                        cropped_face, ml_model, unknown_thresh
                    )

                    if pred is None:
                        st.error("Could not extract embedding from the detected face.")
                    elif pred == "unknown":
                        best_guess = classes[np.argmax(probs)].replace('_', ' ').title()
                        st.markdown(f"""<div class='unknown-card'>
                            <h2 style='color:#ff6b6b'>🔴 Unknown Person</h2>
                            <p>Best guess: <b>{best_guess}</b> ({conf*100:.1f}%) — below threshold ({unknown_thresh*100:.0f}%)</p>
                            <small>Lower the threshold slider or add this person to the dataset.</small>
                        </div>""", unsafe_allow_html=True)
                    else:
                        pretty = pred.replace('_', ' ').title()
                        st.markdown(f"""<div class='result-card'>
                            <p style='color:#888;margin:0;font-size:0.9rem'>IDENTIFIED AS</p>
                            <div class='person-name'>{pretty}</div>
                            <div class='conf-badge'>Confidence: {conf*100:.1f}%</div>
                        </div>""", unsafe_allow_html=True)
                        st.balloons()

                    # ── Top probability bars ────────────────────────────
                    if probs is not None:
                        st.markdown("<br>**Top predictions (all classes):**", unsafe_allow_html=True)
                        top_indices = np.argsort(probs)[::-1][:min(7, len(classes))]
                        for idx in top_indices:
                            pct   = float(probs[idx]) * 100
                            name  = classes[idx].replace('_', ' ').title()
                            is_top = (classes[idx] == (pred if pred != "unknown" else classes[np.argmax(probs)]))
                            bar_color = "#00d4ff" if is_top else "#2a2a4a"
                            txt_color = "#fff" if is_top else "#888"
                            st.markdown(f"""
                            <div style='margin:5px 0;display:flex;align-items:center;gap:10px'>
                              <span style='width:180px;flex-shrink:0;color:{txt_color};font-weight:{"700" if is_top else "400"}'>{name}</span>
                              <div style='flex:1;background:#111;border-radius:4px;height:16px;position:relative'>
                                <div style='width:{max(pct,0.5):.1f}%;background:{bar_color};
                                            height:16px;border-radius:4px;transition:width 0.4s'></div>
                              </div>
                              <span style='color:{txt_color};width:55px;text-align:right;font-weight:{"700" if is_top else "400"}'>{pct:.1f}%</span>
                            </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # ── Test Another Image button ─────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Test Another Image", key="another_btn"):
            st.session_state.uploader_key += 1
            st.rerun()

else:
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;
                border:2px dashed rgba(0,212,255,0.2);
                border-radius:20px;color:#555;margin-top:20px'>
        <h2 style='color:#444'>📂 Upload an Image or Use Webcam to Begin</h2>
        <p>Supports file uploads (JPG, PNG, WEBP) or live webcam capture</p>
        <p style='color:#333;font-size:0.85rem'>The model recognizes: Albert Einstein, Cristiano Ronaldo, Elon Musk,<br>
        Michael Jackson, Nelson Mandela, PM Dr. Abiy, President Trump,<br>
        Robert Oppenheimer, Seble, Zach</p>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET GALLERY
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📂 Training Dataset Gallery")
st.markdown("Browse the training images used to build the recognition model:")

if class_names:
    gallery_cols = st.columns([1, 4])
    with gallery_cols[0]:
        selected_person = st.selectbox(
            "Select Person:",
            class_names,
            format_func=lambda x: x.replace('_', ' ').title()
        )

    with gallery_cols[1]:
        if selected_person:
            person_path = os.path.join(DATA_DIR, selected_person)
            if os.path.exists(person_path):
                images = [f for f in os.listdir(person_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    n_show = min(len(images), 6)
                    img_cols = st.columns(n_show)
                    for idx, img_name in enumerate(images[:n_show]):
                        with img_cols[idx]:
                            img_full_path = os.path.join(person_path, img_name)
                            st.image(img_full_path, caption=f"Sample {idx+1}", width=110)
                    st.caption(f"Showing {n_show} of {len(images)} training images for {selected_person.replace('_',' ').title()}")
                else:
                    st.info("No training images found for this person.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#333;font-size:0.8rem'>"
    "Deep Learning Face Recognition · FaceNet + SVM · Streamlit · MediaPipe</p>",
    unsafe_allow_html=True
)
