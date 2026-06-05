import json
import os

NOTEBOOK_PATH = "face_recognition.ipynb"

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []

new_cells.append(create_markdown_cell("""## 1. Advanced Machine Learning Pipeline & Hyperparameter Tuning
In this section, we extract raw FaceNet embeddings for all images in our cleaned dataset and build Machine Learning classifiers (SVM and KNN) on top of them."""))

new_cells.append(create_code_cell("""import pandas as pd
import numpy as np
from deepface import DeepFace
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib
import time
import os

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
"""))

new_cells.append(create_markdown_cell("""## 2. SVM Hyperparameter Tuning & Training
We will train a Support Vector Machine (SVM) and tune its `C` parameter."""))

new_cells.append(create_code_cell("""# Split dataset (using a small test_size or stratified k-fold since the dataset is small)
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
"""))

new_cells.append(create_markdown_cell("""## 3. KNN Hyperparameter Tuning & Analysis
Let's analyze how the K parameter affects our KNN model."""))

new_cells.append(create_code_cell("""print("Tuning KNN Classifier...")
knn_accuracies = []
k_values = [1, 2, 3, 4, 5]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    knn_accuracies.append(acc)

# Plot hyperparameter graph using Plotly
fig_knn = px.line(x=k_values, y=knn_accuracies, markers=True, 
                  title="KNN Accuracy vs K Value", 
                  labels={'x':'K (Number of Neighbors)', 'y':'Accuracy'})
fig_knn.show()

best_k = k_values[np.argmax(knn_accuracies)]
best_knn = KNeighborsClassifier(n_neighbors=best_k, metric='cosine')
best_knn.fit(X_train, y_train)
print(f"Best KNN Model has K={best_k} with accuracy {max(knn_accuracies)*100:.2f}%")
"""))

new_cells.append(create_markdown_cell("""## 4. Inference Time Comparison
Let's compare the prediction speed of SVM vs KNN."""))

new_cells.append(create_code_cell("""# Inference time test
test_sample = X_test[0].reshape(1, -1)

# SVM Time
start = time.time()
best_svm.predict(test_sample)
svm_time = (time.time() - start) * 1000 # ms

# KNN Time
start = time.time()
best_knn.predict(test_sample)
knn_time = (time.time() - start) * 1000 # ms

fig_time = px.bar(x=['SVM', 'KNN'], y=[svm_time, knn_time], 
                  color=['SVM', 'KNN'],
                  title="Inference Time Comparison (milliseconds)",
                  labels={'x': 'Model', 'y': 'Time (ms)'})
fig_time.show()
"""))

new_cells.append(create_markdown_cell("""## 5. Exporting the Best Model for the Streamlit App
We will save the SVM model since SVMs generally perform exceptionally well on FaceNet embeddings."""))

new_cells.append(create_code_cell("""model_export_path = "best_model.pkl"
joblib.dump(best_svm, model_export_path)
print(f"Saved the best model to {model_export_path} for production deployment!")
"""))

new_cells.append(create_markdown_cell("""## 6. PCA and t-SNE Clustering Visualizations
Dimensionality reduction allows us to visualize the 128-dimensional embedding space in 2D and 3D."""))

new_cells.append(create_code_cell("""print("Performing PCA...")
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['Person'] = y

fig_pca_2d = px.scatter(df_pca, x='PC1', y='PC2', color='Person', 
                        title="2D PCA of FaceNet Embeddings",
                        hover_data=['Person'])
fig_pca_2d.show()

fig_pca_3d = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Person', 
                           title="3D PCA of FaceNet Embeddings")
fig_pca_3d.update_traces(marker=dict(size=4))
fig_pca_3d.show()

print("Performing t-SNE...")
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
X_tsne = tsne.fit_transform(X)

df_tsne = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
df_tsne['Person'] = y

fig_tsne = px.scatter(df_tsne, x='Dim1', y='Dim2', color='Person', 
                      title="2D t-SNE of FaceNet Embeddings")
fig_tsne.show()
"""))

new_cells.append(create_markdown_cell("""## 7. Confusion Matrix Heatmap
Let's look at where the SVM model makes errors."""))

new_cells.append(create_code_cell("""import plotly.figure_factory as ff

cm = confusion_matrix(y_test, svm_pred, labels=class_names)
fig_cm = ff.create_annotated_heatmap(z=cm, x=class_names, y=class_names, colorscale='Blues')
fig_cm.update_layout(title_text='SVM Confusion Matrix Heatmap',
                     xaxis_title="Predicted Label",
                     yaxis_title="True Label")
fig_cm.show()
"""))

nb['cells'].extend(new_cells)

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully with ML pipelines and Plotly visualizations!")
