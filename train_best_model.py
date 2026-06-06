"""
train_best_model.py
===================
Trains the best possible face recognition model using:
- FaceNet embeddings (DeepFace)
- L2 normalization
- GridSearchCV to find best SVM parameters
- Saves to best_model.pkl
"""
import sys, os, logging
logging.getLogger("deepface").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from deepface import DeepFace
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

DATA_DIR         = 'dataset_cleaned'
MODEL_EXPORT     = 'best_model.pkl'
FACENET_MODEL    = 'Facenet'

print("=" * 60)
print("  TRAINING BEST FACE RECOGNITION MODEL")
print("=" * 60)

# ── Step 1: Extract embeddings ──────────────────────────────
class_names = sorted([c for c in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, c))])
print(f"\n[1/4] Found {len(class_names)} people: {class_names}")

X, y = [], []

for person in class_names:
    person_path = os.path.join(DATA_DIR, person)
    imgs = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    person_count = 0
    for img_name in imgs:
        img_path = os.path.join(person_path, img_name)
        try:
            objs = DeepFace.represent(
                img_path=img_path,
                model_name=FACENET_MODEL,
                enforce_detection=False
            )
            if objs:
                emb = objs[0]['embedding']
                X.append(emb)
                y.append(person)
                person_count += 1
        except Exception as e:
            print(f"  [!] Skipping {img_path}: {e}")
    print(f"  -> {person}: {person_count} embeddings")

X = np.array(X)
y = np.array(y)
print(f"\n  Total: {len(X)} embeddings, {len(class_names)} classes")

if len(X) == 0:
    print("ERROR: No embeddings extracted. Check dataset_cleaned.")
    sys.exit(1)

# ── Step 2: L2 Normalize ────────────────────────────────────
print("\n[2/4] Normalizing embeddings (L2)...")
norms = np.linalg.norm(X, axis=1, keepdims=True)
X_norm = X / np.where(norms == 0, 1, norms)
print(f"  Embedding shape: {X_norm.shape}")

# ── Step 3: Grid Search for best SVM ───────────────────────
print("\n[3/4] Searching for best SVM hyperparameters...")

# Use StratifiedKFold — handles small class sizes
n_splits = min(5, min([sum(y == c) for c in class_names]))
n_splits = max(2, n_splits)  # at least 2
print(f"  Using {n_splits}-fold cross-validation")

param_grid = [
    {'kernel': ['linear'], 'C': [0.1, 1.0, 10.0]},
    {'kernel': ['rbf'],    'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']},
]

svc = SVC(probability=True, class_weight='balanced', random_state=42)
cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

grid = GridSearchCV(
    svc, param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_norm, y)

best_model  = grid.best_estimator_
best_params = grid.best_params_
best_cv_acc = grid.best_score_

print(f"\n  Best params: {best_params}")
print(f"  Best CV accuracy: {best_cv_acc*100:.2f}%")

# ── Step 4: Retrain on full dataset & evaluate ──────────────
print("\n[4/4] Retraining best model on full dataset...")
best_model.fit(X_norm, y)

train_preds = best_model.predict(X_norm)
train_acc   = accuracy_score(y, train_preds)
print(f"  Training accuracy: {train_acc*100:.2f}%")
print(f"\n  Classification Report:\n")
print(classification_report(y, train_preds, target_names=class_names))

# ── Save model ──────────────────────────────────────────────
joblib.dump(best_model, MODEL_EXPORT)
print(f"\n{'='*60}")
print(f"  Model saved to: {MODEL_EXPORT}")
print(f"  Kernel: {best_params.get('kernel')}, C={best_params.get('C')}")
print(f"  CV Accuracy: {best_cv_acc*100:.2f}%")
print(f"  Train Accuracy: {train_acc*100:.2f}%")
print(f"  Classes: {list(best_model.classes_)}")
print(f"{'='*60}")
