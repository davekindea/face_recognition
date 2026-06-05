import streamlit as st
import cv2
import os
import sys
import mediapipe as mp
import urllib.request
import numpy as np
from deepface import DeepFace
from PIL import Image
import joblib

# Set page configuration
st.set_page_config(
    page_title="Deep Learning Face Recognition",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    /* Global Background and Fonts */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d2ff;
        font-weight: 700;
        text-shadow: 0px 2px 10px rgba(0, 210, 255, 0.3);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(30, 30, 47, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Custom Card for Results */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
    }
    
    /* Image Styling */
    img {
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    
    /* Confidence Badge */
    .confidence-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        font-weight: bold;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
IMG_SIZE = (224, 224)
MODEL_PATH = os.path.join(os.getcwd(), 'blaze_face_short_range.tflite')
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
BEST_MODEL_FILE = 'best_model.pkl'

# Ensure face detector model is downloaded
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading MediaPipe face detector model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Initialize MediaPipe Face Detector
@st.cache_resource
def get_face_detector(model_path, confidence):
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        min_detection_confidence=confidence
    )
    return FaceDetector.create_from_options(options)

# Helper function to detect and crop face
def crop_face(img_np, detector):
    h, w = img_np.shape[:2]
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = detector.detect(mp_image)
    
    if detection_result.detections:
        bbox = detection_result.detections[0].bounding_box
        x1 = max(0, int(bbox.origin_x))
        y1 = max(0, int(bbox.origin_y))
        x2 = min(w, int(bbox.origin_x + bbox.width))
        y2 = min(h, int(bbox.origin_y + bbox.height))
        
        face_crop = img_np[y1:y2, x1:x2]
        if face_crop.size > 0:
            face_resized = cv2.resize(face_crop, IMG_SIZE)
            return face_resized, (x1, y1, x2, y2)
            
    return None, None

# App Layout
st.markdown("<h1>🤖 Deep Learning Face Predictor</h1>", unsafe_allow_html=True)
st.markdown("### Upload an image to identify the person using state-of-the-art Deep Learning (FaceNet) and Machine Learning (SVM).")

# Sidebar
st.sidebar.markdown("<h2>⚙️ Settings</h2>", unsafe_allow_html=True)
min_confidence = st.sidebar.slider("Face Detection Confidence", 0.10, 1.00, 0.50, 0.05)

# Load ML Model
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Model Status")
ml_model = None

def train_model():
    import pandas as pd
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    
    DATA_DIR = 'dataset_cleaned'
    MODEL_NAME = 'Facenet'
    
    st.sidebar.info("Extracting embeddings... This will take a few minutes. Please wait.")
    data = []
    labels = []
    class_names = sorted([c for c in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, c))])
    
    for person in class_names:
        person_path = os.path.join(DATA_DIR, person)
        for img_name in os.listdir(person_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            img_path = os.path.join(person_path, img_name)
            try:
                embedding_objs = DeepFace.represent(img_path=img_path, model_name=MODEL_NAME, enforce_detection=False)
                if embedding_objs:
                    data.append(embedding_objs[0]['embedding'])
                    labels.append(person)
            except Exception as e:
                pass
                
    X = np.array(data)
    y = np.array(labels)
    
    if len(X) > 0:
        st.sidebar.info("Training SVM Classifier...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        svm_grid = GridSearchCV(SVC(probability=True, kernel='linear'), {'C': [1]}, cv=3)
        svm_grid.fit(X_train, y_train)
        best_svm = svm_grid.best_estimator_
        joblib.dump(best_svm, BEST_MODEL_FILE)
        return True
    return False

try:
    if os.path.exists(BEST_MODEL_FILE):
        ml_model = joblib.load(BEST_MODEL_FILE)
        st.sidebar.success(f"✅ Loaded ML Model: `{type(ml_model).__name__}`")
    else:
        st.sidebar.error("❌ `best_model.pkl` not found!")
        st.sidebar.warning("Model needs to be trained on the cloud server.")
        if st.sidebar.button("Train Model Now"):
            with st.spinner("Training model on the cloud..."):
                if train_model():
                    st.sidebar.success("Training complete! Please refresh the page.")
                    st.rerun()
                else:
                    st.sidebar.error("Training failed.")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# Main Content
uploaded_file = st.file_uploader("Upload a face image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    with col1:
        st.markdown("<h3 style='text-align: center;'>📸 Input Image</h3>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with col2:
        st.markdown("<h3 style='text-align: center;'>🔍 Analysis & Prediction</h3>", unsafe_allow_html=True)
        
        detector = get_face_detector(MODEL_PATH, min_confidence)
        cropped_face, bbox = crop_face(image, detector)
        
        if cropped_face is not None:
            # Display cropped face
            cropped_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            st.image(cropped_rgb, caption="Detected Face (224x224)", width=150)
            
            if ml_model is not None:
                temp_crop_path = "temp_crop.jpg"
                cv2.imwrite(temp_crop_path, cropped_face)
                
                with st.spinner("Extracting Deep Learning Embeddings (FaceNet)..."):
                    try:
                        # Extract embedding using DeepFace
                        embedding_objs = DeepFace.represent(img_path=temp_crop_path, model_name='Facenet', enforce_detection=False)
                        if embedding_objs:
                            embedding = np.array(embedding_objs[0]['embedding']).reshape(1, -1)
                            
                            # ML Prediction
                            prediction = ml_model.predict(embedding)[0]
                            
                            # Get probability if model supports it
                            if hasattr(ml_model, "predict_proba"):
                                probabilities = ml_model.predict_proba(embedding)[0]
                                confidence = np.max(probabilities) * 100
                            else:
                                confidence = 100.0
                                
                            st.markdown(f"""
                            <div class="result-card">
                                <h2>Identity Confirmed</h2>
                                <h1 style="color: #00ff88; font-size: 2.5rem; margin: 10px 0;">{prediction.replace('_', ' ').title()}</h1>
                                <div class="confidence-badge">Confidence: {confidence:.2f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                            
                    except Exception as df_err:
                        st.error(f"Error extracting embedding: {df_err}")
                
                if os.path.exists(temp_crop_path):
                    os.remove(temp_crop_path)
            else:
                st.warning("⚠️ Cannot predict identity: ML Model not found. Please run the Jupyter Notebook first.")
        else:
            st.error("❌ No face detected. Try another photo or lower the confidence threshold.")
else:
    st.info("👆 Upload an image to start the facial recognition pipeline.")
