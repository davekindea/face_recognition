import sys
import os

# Fix Windows terminal encoding (emoji/unicode support)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

import numpy as np
from deepface import DeepFace
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import logging
logging.getLogger("deepface").setLevel(logging.ERROR)

DATA_DIR = 'dataset_cleaned'
MODEL_NAME = 'Facenet'
MODEL_EXPORT_PATH = "best_model.pkl"

print("Extracting/Loading embeddings for the dataset...")
data = []
labels = []
class_names = sorted([c for c in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, c))])

for person in class_names:
    person_path = os.path.join(DATA_DIR, person)
    for img_name in os.listdir(person_path):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        img_path = os.path.join(person_path, img_name)
        try:
            # Extract 128D FaceNet embedding
            embedding_objs = DeepFace.represent(img_path=img_path, model_name=MODEL_NAME, enforce_detection=False)
            if embedding_objs:
                embedding = embedding_objs[0]['embedding']
                data.append(embedding)
                labels.append(person)
        except Exception as e:
            print(f"Error on {img_path}: {e}")

X = np.array(data)
y = np.array(labels)
print(f"Extracted {len(X)} embeddings for {len(class_names)} people.")

if len(X) > 0:
    # L2 normalize embeddings
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    print("Training Linear SVM on the entire dataset...")
    best_svm = SVC(probability=True, kernel='linear', C=1.0)
    best_svm.fit(X_norm, y)
    
    # Calculate training accuracy
    train_preds = best_svm.predict(X_norm)
    train_acc = accuracy_score(y, train_preds)
    print(f"Linear SVM Training Accuracy: {train_acc*100:.2f}%")
    
    joblib.dump(best_svm, MODEL_EXPORT_PATH)
    print(f"Saved the best model to {MODEL_EXPORT_PATH} for production deployment!")
else:
    print("No data extracted. Ensure dataset_cleaned has images.")
