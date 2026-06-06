import joblib, numpy as np, os
from deepface import DeepFace
import logging
logging.getLogger('deepface').setLevel(logging.ERROR)

model = joblib.load('best_model.pkl')
print('Classes:', list(model.classes_))

DATA_DIR = 'dataset_cleaned'
MODEL_NAME = 'Facenet'

correct = 0
total = 0

for person in sorted(os.listdir(DATA_DIR)):
    person_path = os.path.join(DATA_DIR, person)
    if not os.path.isdir(person_path):
        continue
    imgs = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for img_name in imgs[:3]:  # test first 3 images per person
        img_path = os.path.join(person_path, img_name)
        try:
            objs = DeepFace.represent(img_path=img_path, model_name=MODEL_NAME, enforce_detection=False)
            if objs:
                emb = np.array(objs[0]['embedding']).reshape(1,-1)
                emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                probs = model.predict_proba(emb_norm)[0]
                pred = model.classes_[np.argmax(probs)]
                conf = probs[np.argmax(probs)]
                match = (pred == person)
                if match:
                    correct += 1
                total += 1
                status = "OK" if match else "WRONG"
                print(f"  [{status}] {person} -> predicted: {pred} ({conf*100:.1f}%)")
        except Exception as e:
            print(f"  Error on {img_path}: {e}")

if total > 0:
    print(f"\nAccuracy on sample: {correct}/{total} = {correct/total*100:.1f}%")
else:
    print("No images tested!")
