# Waste Image Classification Hackathon 2025

## Introduction

Waste Image Classification Hackathon 2025 challenged us to develop an AI-based system capable of automatically classifying waste images into two categories: **Organic Waste** and **Recyclable Waste**. 
Our objective was to build a complete pipeline—from data preprocessing, feature extraction, model training, and evaluation to deployment and UI integration—aimed at improving waste-sorting efficiency and promoting environmental sustainability. We trained and evaluated our model locally on our PC and developed a Flask API to integrate with a responsive waste classification website.


---

## Data Preprocessing

### Overview

In this stage, we loaded the raw dataset (downloaded from [this link](https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/n3gtgm9jxj-2.zip)), then cleaned, standardized, and normalized the images. We used **Google Colab** for initial processing and our local PC for training. The tools and libraries we used include:

- **Python & Jupyter/Colab Notebooks:** For scripting and data exploration.
- **PIL & OpenCV:** For image loading, cleaning, and resizing.
- **NumPy:** For array manipulation and normalization.

### What We Did

1. **Cleaning:**  
   We removed corrupt images by verifying each file with PIL.

2. **Standardization:**  
   All images were resized to **256*256 pixels** to ensure consistency across the dataset.

3. **Normalization:**  
   Pixel values were normalized to the range **[0, 1]** to facilitate efficient training.

### Code Snippet


```python
import numpy as np
import os
import zipfile
import shutil  # Added for deletion
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import random

# Define Paths
zip_path = "/content/Waste Classification Dataset.zip"
extract_path = "/content/Waste_Classification"
saved_images_dir = "/content/Processed Data/preprocessed_batches"
saved_labels_path = "/content/Processed Data/preprocessed_labels.npy"

# Force Re-Extraction: Delete old extracted folder and re-extract dataset
if os.path.exists(extract_path):
    print("Deleting old extracted dataset...")
    shutil.rmtree(extract_path)  # Delete the extracted folder

print("Extracting dataset...")
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_path)
print("Extraction complete!")

# Re-check total images AFTER extraction
total_images = sum([len(files) for _, _, files in os.walk(extract_path) if any(file.endswith((".jpg")) for file in files)])
print(f"Total extracted images: {total_images}")

# Load Image Paths & Labels
labels, img_paths = [], []
for root, dirs, files in os.walk(extract_path):
    category = os.path.basename(root)
    if category in ["recyclable", "organic"]:
        for file in files:
            if file.lower().endswith((".jpg")):
                labels.append(category)
                img_paths.append(os.path.join(root, file))

print(f"Total images found: {len(img_paths)}")
print(f"Total labels found: {len(labels)}")
assert len(img_paths) == len(labels), "Mismatch between images and labels!"

# Encode Labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Ensure Directory Exists Before Saving Labels
os.makedirs(os.path.dirname(saved_labels_path), exist_ok=True)

# Save Labels to Drive
np.save(saved_labels_path, y)
print(f"Saved {len(y)} labels successfully.")

# DELETE OLD PREPROCESSED BATCHES BEFORE PROCESSING NEW ONES
if os.path.exists(saved_images_dir):
    print("Deleting old preprocessed image batches...")
    shutil.rmtree(saved_images_dir)

# Standardization Function
def standardize_images(images):
    """Standardize images: subtract mean and divide by standard deviation."""
    mean = np.mean(images, axis=(0, 1, 2), keepdims=True)
    std = np.std(images, axis=(0, 1, 2), keepdims=True)
    return (images - mean) / (std + 1e-7)  # Avoid division by zero

# Batch Processing Function
def preprocess_images(image_paths, batch_size=500):
    """Process images in batches and save them separately to prevent memory overload."""
    os.makedirs(saved_images_dir, exist_ok=True)  # Ensure directory exists

    for i in range(0, len(image_paths), batch_size):
        batch_file_path = os.path.join(saved_images_dir, f"batch_{i // batch_size}.npy")

        images = []
        for img_path in image_paths[i : i + batch_size]:  # Process batch
            try:
                img = Image.open(img_path).convert("RGB").resize((256, 256))
                img = np.array(img, dtype=np.float32) / 255.0  # Normalize
                images.append(img)
            except Exception as e:
                print(f"⚠ Skipping corrupted image: {img_path}")

        batch = np.array(images)
        np.save(batch_file_path, batch)  # Save batch
        print(f"Saved batch {i // batch_size}: {batch.shape}")

print("Preprocessing images in batches...")
preprocess_images(img_paths, batch_size=500)
print("Image preprocessing complete!")
```


---
## Feature Extraction and Model Training

### Overview

After preprocessing, we extracted features from the images and split the data into training and testing batches.
For feature extraction, each image was loaded from the saved batches, reshaped (flattened) for model compatibility, and normalized. We then performed an 80/20 split using train_test_split to ensure reproducibility and balanced class distribution.
We used SGDClassifier for incremental training of our model with class weights to address any class imbalance.

## Data splitting

After feature extraction, we split the dataset into training (80%) and testing (20%) batches.
Rather than loading the entire dataset into memory (which can cause memory issues), we processed and saved the data in batches using NumPy, then used scikit-learn’s train_test_split to generate indices for training and testing, ensuring a stratified split.

### Code Snippet for feature extraction and Model training

```python
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import SGDClassifier
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

saved_images_dir = "/content/Processed Data/preprocessed_batches"
saved_labels_path = "/content/Processed Data/preprocessed_labels.npy"

# Load labels
y = np.load(saved_labels_path)

# Compute class weights to address the imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))

# Train-test split
train_indices, test_indices = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)

# Initialize SGDClassifier model with class weights
svm_model = SGDClassifier(loss="log_loss", random_state=42, early_stopping=False, class_weight=class_weight_dict)

# Precompute Standard Scaler
scaler = StandardScaler()
all_X_train = []

# Fit scaler on all training data first
num_batches = 10  # Assume we know the number of batches
batch_size = 500  # Define batch size before use
for batch_index in range(num_batches):
    batch_path = os.path.join(saved_images_dir, f"batch_{batch_index}.npy")
    if os.path.exists(batch_path):
        X_batch = np.load(batch_path)
        X_batch = X_batch.reshape(X_batch.shape[0], -1)
        batch_indices = np.arange(batch_index * batch_size, batch_index * batch_size + len(X_batch))
        train_mask = np.isin(batch_indices, train_indices)
        all_X_train.append(X_batch[train_mask])

all_X_train = np.vstack(all_X_train)
scaler.fit(all_X_train)  # Fit scaler once

# Train in batches
print("Training SVM model in batches...")
training_accuracies = []
batch_index = 0
while True:
    batch_path = os.path.join(saved_images_dir, f"batch_{batch_index}.npy")
    if not os.path.exists(batch_path):
        break  # No more batches

    X_batch = np.load(batch_path)
    X_batch = X_batch.reshape(X_batch.shape[0], -1)
    batch_indices = np.arange(batch_index * batch_size, batch_index * batch_size + len(X_batch))
    train_mask = np.isin(batch_indices, train_indices)

    # Select training data
    X_train_batch = X_batch[train_mask]
    y_train_batch = y[train_indices[np.isin(train_indices, batch_indices)]]

    if len(X_train_batch) == 0:
        batch_index += 1
        continue  # Skip empty batches

    # Scale data
    X_train_batch = scaler.transform(X_train_batch)

    # Incrementally train model without class_weight in partial_fit
    svm_model.partial_fit(X_train_batch, y_train_batch, classes=np.unique(y))

    # Predict on training batch only
    y_train_pred = svm_model.predict(X_train_batch)

    # Compute accuracy
    batch_accuracy = accuracy_score(y_train_batch, y_train_pred)
    training_accuracies.append(batch_accuracy)

    batch_index += 1

print("SVM training complete!")

import joblib
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```
---

## Model Evaluation

### Overview
We evaluated our trained model on the test set using the saved preprocessed images and labels. For memory efficiency, the evaluation code loads test batches one by one, makes predictions, and aggregates results before computing evaluation metrics.

### Code
```python
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Define the test batches directory
test_dir = "test_batches"

# Get sorted list of test batch files (assuming naming convention "X_test_batch_*.npy" and "y_test_batch_*.npy")
test_feature_files = sorted([f for f in os.listdir(test_dir) if f.startswith("X_test")])
test_label_files   = sorted([f for f in os.listdir(test_dir) if f.startswith("y_test")])

# Load the trained model
model = joblib.load("svm_model.pkl")

y_true_all = []
y_pred_all = []

print("Starting evaluation on test batches...")
# Process each test batch one at a time
for i, (feat_file, label_file) in enumerate(zip(test_feature_files, test_label_files)):
    X_batch = np.load(os.path.join(test_dir, feat_file))
    y_batch = np.load(os.path.join(test_dir, label_file))
    y_pred_batch = model.predict(X_batch)
    y_true_all.append(y_batch)
    y_pred_all.append(y_pred_batch)
    print(f"Processed test batch {i+1}/{len(test_feature_files)}")

y_true = np.hstack(y_true_all)
y_pred = np.hstack(y_pred_all)

accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["Recyclable", "Organic"])

print("\nEvaluation Results:")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)
```
We attained an accuracy of 76.95%

![evaluation](https://github.com/user-attachments/assets/02db3d63-a25d-4ecf-9933-dd80b30c8e69)


## Deployment & API

### Overview

To integrate our trained waste classification model into a real-world application, we developed a Flask API that serves predictions in real time.
The API accepts an image upload, preprocesses the image (converting it to RGB, normalizing, and resizing to (256, 256)), and then runs it through the model to return the classification result along with a URL to display the uploaded image.

### Key Functionalities of the API

Image Upload – Users can upload a waste image for classification.

Preprocessing – The uploaded image is resized, converted to grayscale, and flattened before being passed to the model.

Model Inference – The trained model predicts whether the waste is recyclable or organic.

Response with Tips – The API returns the prediction along with actionable waste disposal tips.

### API Endpoints

The API provides a single main endpoint:

/predict (POST)

Accepts an image file.

Preprocesses the image.

Loads the trained model.

Returns the classification result and waste disposal recommendations.

### Code

```python
from flask import Flask, render_template, request, send_from_directory
import numpy as np
import os
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model_path = os.path.join("model_processing", "model.keras")

try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Define model even if loading fails

# Define image preprocessing function
def preprocess_image(filepath):
    """Loads and preprocesses the image for model inference."""
    image = Image.open(filepath).convert('RGB')
    image = np.array(image, dtype=np.float32) / 255.0
    image = tf.image.resize(image, (128, 128))
    image = np.expand_dims(image, axis=0)
    return image

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# About Page
@app.route('/about')
def about():
    return render_template('about.html')

# Contact Page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Classification Page
@app.route('/classify')
def classify():
    return render_template('classify.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('classify.html', prediction="No file uploaded")
        file = request.files['file']
        if file.filename == '':
            return render_template('classify.html', prediction="No selected file")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        print(f"Image saved at: {filepath}")
        image_array = preprocess_image(filepath)
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        class_names = ['Organic', 'Recyclable']
        predicted_class = class_names[predicted_class]
        return render_template('classify.html', prediction=predicted_class, image_url=f'/uploads/{file.filename}')
    except Exception as e:
        return render_template('classify.html', prediction=f"Error: {str(e)}")

# Serve Uploaded Images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
```

## UI Integration

### Overview

Our waste classification website is built using HTML, CSS, and JavaScript.
It features:

A responsive navigation bar.

A full-window Swiper slider on the homepage.

A classification page where users can upload an image and see the prediction and waste handling tips.

Additional pages for model information (About) and developer profiles (Contact).

 
#### Screenshots:

##### Home page  

![Home](https://github.com/user-attachments/assets/77f24ae7-4bf0-4492-98c7-443ddada9579)

##### Classify page

![classify](https://github.com/user-attachments/assets/454332d4-8cb7-4679-bbe6-ee62b4bd18bc)

##### About page

![about (2)](https://github.com/user-attachments/assets/a6b4c5d3-3128-438d-8d25-fc7d16aad1a7)

##### Contact page

![contact](https://github.com/user-attachments/assets/9243b397-0220-4fe1-8167-0c50c8255e5a)


## Classified Images from our model

![pred](https://github.com/user-attachments/assets/a4fab64a-ffdd-4278-827a-bf9c0027f1c4)

![pred2](https://github.com/user-attachments/assets/ae980885-5962-4d20-8810-71ecbfcabded)

## Conclusion

This project successfully developed an AI-based waste classification system that:

Automates waste sorting into organic and recyclable categories.

Improves efficiency through robust data preprocessing, feature extraction, and incremental model training.

Integrates a responsive web interface with a Flask API for real-time predictions.

### Key Findings and Impact

Model Performance:
The logistic regression classifier, trained incrementally on a large, preprocessed dataset, demonstrated promising accuracy and reliability.

Real-World Impact:
Automated waste classification can greatly improve recycling efficiency, reduce landfill usage, and promote environmental sustainability.

### Challenges Faced

Managing memory constraints during image preprocessing.

Ensuring robust data cleaning and handling diverse image formats.

Integrating model predictions with a responsive web interface.

Overcoming occasional misclassifications.




---
✨ Developed by: Angel Kuria & Karanja Nyambura
