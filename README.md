# Waste Image Classification Hackathon 2025: Smart Waste Management

**University of Eastern Africa, Baraton**  
**Department of Information Systems and Computing**  
**P.O BOX 2500-30100 Eldoret, Kenya**

---

## Introduction

Waste Image Classification Hackathon 2025 is a hands-on challenge where our team developed an AI-based model to automatically classify waste images into two categories: **Organic Waste** and **Recyclable Waste**. The project’s objective is to improve waste-sorting efficiency and promote environmental sustainability through automated waste classification. Our solution combines data preprocessing, model training using Logistic Regression, thorough model evaluation, and full-stack integration with a responsive web interface that serves model predictions via a Flask API.

---

## Data Preprocessing

### Data Sourcing and Exploration

We downloaded the waste image dataset from [this link](https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/n3gtgm9jxj-2.zip). Using Google Colab and Python, we explored the dataset in our `Data_Preprocessing.ipynb` notebook. In this notebook, we visualized sample images to understand the dataset's diversity and quality, which helped us plan our cleaning and standardization steps.

### Cleaning, Standardization, and Normalization

To prepare the images for model training, we first cleaned the dataset by removing any corrupt images and duplicates. Next, we standardized all images by resizing them to **256×256 pixels** so that every input has a consistent shape. This uniformity is crucial for feeding the images into our model.

After resizing, we normalized the pixel values by scaling them to the range **[0, 1]**. This normalization helps improve the model's performance and convergence during training.

During the preprocessing phase, we used TensorFlow to load images and build a `tf.data` pipeline, allowing us to process the images in efficient batches. We also incorporated OpenCV for additional image processing tasks, such as converting image color channels, while NumPy was indispensable for array manipulation throughout our workflow.

Overall, the combination of Google Colab, Python, TensorFlow, OpenCV, and NumPy enabled us to clean, standardize, and normalize a large and diverse dataset effectively, setting a strong foundation for the subsequent model training and evaluation stages.


### Challenges Faced

- **Memory Constraints:**  
  Handling large datasets required batch processing using the `tf.data.Dataset` API.

- **Data Inconsistencies:**  
  Some images were corrupt or in different formats, necessitating robust error handling in the preprocessing code.

---

## Model Training

### Approach

We opted for a **Logistic Regression** model for its interpretability and ease of deployment. Given the high-dimensional nature of image data (flattened pixels), we used incremental training via scikit-learn’s `SGDClassifier` configured with logistic loss (simulating logistic regression).

### Code Snippets and Explanations

- **Notebook:**  
  See `Feature_Extraction_and_Model.ipynb` (accessible via [Colab Notebook 2](https://colab.research.google.com/drive/1hZEPynEOFGJq9J49D1sxK0cCWkQkz2Od?usp=drive_link)).

- **Data Split:**  
  The dataset was divided using an 80/20 train-test split.

- **Training:**  
  Incremental training was implemented using `partial_fit` to process data in batches, thereby mitigating memory issues.

Example snippet:
```python
from sklearn.linear_model import SGDClassifier
from joblib import dump
import numpy as np

# Initialize SGDClassifier for logistic regression
clf = SGDClassifier(loss='log', max_iter=1, tol=None)
classes_unique = np.unique(labels_np)

# Incremental training over multiple epochs
for epoch in range(num_epochs):
    for batch_data, batch_labels in dataset:
        clf.partial_fit(batch_data.numpy(), batch_labels.numpy(), classes=classes_unique)

# Save model using joblib
dump(clf, 'logistic_regression_model.joblib')


## Model Evaluation

### Metrics
- **Accuracy, Precision, Recall, F1-score:**  
  Computed to assess the classifier’s performance.
- **Confusion Matrix:**  
  Visualized to inspect misclassifications.
- **Visualizations:**  
  ROC curves and heatmaps were generated using matplotlib and seaborn.

### Example Discussion
- Our model achieved an accuracy of **XX%** on the test set.
- The confusion matrix indicated that most misclassifications occurred between similar types of recyclable waste.
- Screenshots of the evaluation plots are included in the repository under the **Evaluation_Screenshots** folder.

---

## Deployment & API Integration

### Deployment
- **Cloud Platform:**  
  The trained model was deployed using **Google Cloud Platform**.
- **API:**  
  A Flask-based API was developed to serve model predictions.

### API Endpoint
- **`/predict`:**  
  Accepts an image file (via POST), processes it, and returns a JSON response with the prediction and waste handling tips.

### Code and Instructions

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
model = joblib.load("logistic_regression_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    file_path = "temp.jpg"
    file.save(file_path)
    # Preprocess the image and make a prediction...
    prediction = model.predict(...)[0]
    return jsonify({"prediction": prediction})


## Deployment & API
The model was deployed using **Flask** to create an API. The endpoint accepts image uploads and returns predictions.

```python
from flask import Flask, request, jsonify
import joblib
import cv2

app = Flask(__name__)
model = joblib.load('waste_classifier_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image_resized = cv2.resize(image, (256, 256)).flatten() / 255.0
    prediction = model.predict([image_resized])
    return jsonify({'prediction': 'Recyclable Waste' if prediction[0] == 1 else 'Organic Waste'})

app.run()
```

## UI Integration

### Overview

#### Website Design:
- A responsive web interface built using **HTML**, **CSS**, and **JavaScript**.

#### Features:
- A full-window **Swiper slider** on the homepage.
- A responsive navigation bar that turns into a dropdown on small screens.
- Dedicated pages for classification, model explanation (About), and developer profiles (Contact).

#### Integration:
- The website communicates with the Flask API to display model predictions in real time.

#### Screenshots:
- Final UI screenshots (Homepage, Classify, About, and Contact pages) are included in the **Screenshots** folder.

## Conclusion

This project successfully developed an AI-based waste classification system that:

- **Automates the sorting of waste images** into organic and recyclable categories.
- **Achieves significant performance improvements** through efficient data preprocessing and incremental model training.
- **Integrates with a user-friendly web interface** and a robust Flask API for real-time predictions.

### Key Findings and Impact

- **Model Performance:**  
  The logistic regression classifier, trained incrementally on a large, preprocessed dataset, demonstrated promising accuracy and reliability.

- **Real-World Impact:**  
  Automated waste classification can greatly improve recycling efficiency, reduce landfill usage, and promote environmental sustainability.

### Challenges Faced

- **Managing memory constraints** during image preprocessing.
- **Ensuring robust data cleaning** and handling diverse image formats.
- **Integrating model predictions** with a responsive web interface.

This project not only highlights technical expertise in machine learning and web development but also underscores the potential for AI-driven solutions to contribute to a cleaner, more sustainable environment.


---
✨ Developed by: Angel Kuria & Karanja Nyambura
