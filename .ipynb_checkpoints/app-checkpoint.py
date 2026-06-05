import streamlit as st
import cv2
import os
import sys
import mediapipe as mp
import urllib.request
import numpy as np
from deepface import DeepFace
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Premium Face Recognition Hub",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stAlert {
        border-radius: 10px;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# App Title & Description
st.title("👤 Premium Face Recognition Hub")
st.markdown("Upload an image to detect the face and identify who it is using the **FaceNet** deep learning model.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
threshold = st.sidebar.slider("Cosine Distance Threshold", min_value=0.10, max_value=1.00, value=0.40, step=0.05,
                              help="Lower values mean stricter matching (fewer false positives). Default is 0.40.")
min_confidence = st.sidebar.slider("Min Detection Confidence", min_value=0.10, max_value=1.00, value=0.50, step=0.05,
                                   help="Confidence threshold for the MediaPipe face detector.")

# Constants
INPUT_DIR  = 'dataset'
DATA_DIR   = 'dataset_cleaned'
IMG_SIZE   = (224, 224)
MODEL_PATH = os.path.join(os.getcwd(), 'blaze_face_short_range.tflite')
MODEL_URL  = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'

# Ensure face detector model is downloaded
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading face detector model..."):
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
    # MediaPipe expects RGB
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

# Main Upload Area
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Show columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with col2:
        st.subheader("🔍 Face Detection & Identification")
        
        # Load detector and find face
        try:
            detector = get_face_detector(MODEL_PATH, min_confidence)
            cropped_face, bbox = crop_face(image, detector)
            
            if cropped_face is not None:
                # Show cropped face
                cropped_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                st.image(cropped_rgb, caption="Detected Face (Resized to 224x224)", width=180)
                
                # Save cropped face to a temporary file for DeepFace search
                temp_crop_path = "temp_crop.jpg"
                cv2.imwrite(temp_crop_path, cropped_face)
                
                # Run DeepFace Search
                with st.spinner("Comparing face with database..."):
                    # Suppress standard output/error to keep streamlit clean
                    sys.stdout = open(os.devnull, 'w')
                    sys.stderr = open(os.devnull, 'w')
                    
                    try:
                        results = DeepFace.find(
                            img_path=temp_crop_path,
                            db_path=DATA_DIR,
                            model_name='Facenet',
                            distance_metric='cosine',
                            enforce_detection=False,
                            silent=True
                        )
                        
                        # Restore stdout/stderr
                        sys.stdout = sys.__stdout__
                        sys.stderr = sys.__stderr__
                        
                        df = results[0] if results else None
                        
                    except Exception as df_err:
                        sys.stdout = sys.__stdout__
                        sys.stderr = sys.__stderr__
                        st.error(f"Error querying DeepFace: {df_err}")
                        df = None
                
                # Clean up temporary cropped file
                if os.path.exists(temp_crop_path):
                    os.remove(temp_crop_path)
                
                if df is not None and not df.empty:
                    # Sort results and get the best match
                    df = df.sort_values('distance').reset_index(drop=True)
                    best_match = df.iloc[0]
                    matched_file_path = best_match['identity']
                    person_name = os.path.basename(os.path.dirname(matched_file_path))
                    distance = best_match['distance']
                    
                    # Convert distance to a similarity percentage (for display)
                    similarity = max(0.0, 1.0 - distance) * 100
                    
                    if distance <= threshold:
                        st.success(f"✅ **Match Found: {person_name.replace('_', ' ').title()}**")
                        
                        # Show match details
                        st.markdown(f"**Similarity Score:** `{similarity:.1f}%` (Cosine Distance: `{distance:.3f}`)")
                        st.progress(similarity / 100.0)
                        
                        # Display side-by-side comparison of cropped face and matched database image
                        match_col1, match_col2 = st.columns(2)
                        with match_col1:
                            st.caption("Uploaded Face")
                            st.image(cropped_rgb, width=150)
                        with match_col2:
                            st.caption(f"Matched Database Image")
                            if os.path.exists(matched_file_path):
                                matched_img = Image.open(matched_file_path)
                                st.image(matched_img, width=150)
                            else:
                                st.warning("Database image not found on disk.")
                                
                        # Show Top 3 matches
                        st.markdown("---")
                        st.write("📊 **Top Matches in Database:**")
                        for idx, row in df.head(3).iterrows():
                            match_name = os.path.basename(os.path.dirname(row['identity'])).replace('_', ' ').title()
                            match_dist = row['distance']
                            match_sim = max(0.0, 1.0 - match_dist) * 100
                            st.write(f"- **{match_name}**: `{match_sim:.1f}%` match (dist: `{match_dist:.3f}`)")
                            
                    else:
                        st.warning(f"⚠️ **Person Unknown**")
                        st.info(f"Closest match was **{person_name.replace('_', ' ').title()}** but similarity was too low (`{similarity:.1f}%` match, limit is `{(1-threshold)*100:.1f}%`).")
                        
                        # Optional: display the closest match even if rejected
                        with st.expander("View closest match details"):
                            st.image(Image.open(matched_file_path), caption=f"Closest: {person_name.replace('_', ' ').title()}", width=150)
                else:
                    st.error("❌ No matches found in the database at all.")
                    
            else:
                st.error("❌ No face detected in the image. Try another photo or adjust the 'Min Detection Confidence' in the sidebar.")
                
        except Exception as e:
            st.error(f"An unexpected error occurred during processing: {e}")
            st.info("Tip: Make sure you've run the Jupyter notebook once to create the cleaned dataset folders and pickle files.")
else:
    st.info("👆 Please upload an image file (JPEG, PNG, WEBP) to get started.")
