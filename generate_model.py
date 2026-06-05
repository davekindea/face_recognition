import pandas as pd
import numpy as np
from deepface import DeepFace
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import logging
logging.getLogger("deepface").setLevel(logging.ERROR)

DATA_DIR = 'dataset_cleaned'
MODEL_NAME = 'Facenet'

print("Extracting embeddings for the dataset...")
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Tuning SVM Classifier...")
    svm_param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
    svm_grid = GridSearchCV(SVC(probability=True), svm_param_grid, cv=3)
    svm_grid.fit(X_train, y_train)
    
    best_svm = svm_grid.best_estimator_
    print(f"Best SVM Parameters: {svm_grid.best_params_}")
    
    svm_pred = best_svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f"SVM Test Accuracy: {svm_acc*100:.2f}%")
    
    model_export_path = "best_model.pkl"
    joblib.dump(best_svm, model_export_path)
    print(f"Saved the best model to {model_export_path} for production deployment!")
else:
    print("No data extracted. Ensure dataset_cleaned has images.")
