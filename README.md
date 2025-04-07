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
from PIL import Image
import os

# Define image cleaning function
def clean_images(image_folder):
    for filename in os.listdir(image_folder):
        try:
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path)
            img.verify()  # Verify if the image is corrupted
        except (IOError, SyntaxError) as e:
            print(f"Removing corrupt image: {filename}")
            os.remove(img_path)
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
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
train_indices, test_indices = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
```
---

## Model Evaluation

### Overview
We evaluated our trained model on the test set using saved preprocessed images and labels. The model achieved an accuracy of 76.95%, with results displayed in confusion matrices and classification reports.

### Evaluation Metrics
1. Accuracy Score
2. Confusion Matrix
3. Classification Report

### Code
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate predictions and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("Classification Report:\n", classification_report(y_true, y_pred))
```
### Output

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

/predict (POST): Accepts an image file and returns the classification result.

### Code

```python
from flask import Flask, request

@app.route('/predict', methods=['POST'])
def predict():
    # Handle file upload and prediction
    ...
```

## UI Integration

### Overview

Our waste classification website is built using HTML, CSS, and JavaScript.
It features:

* A responsive navigation bar.

* A full-window Swiper slider on the homepage.

* A classification page where users can upload an image and see the prediction and waste handling tips.

* Additional pages for model information (About) and developer profiles (Contact).

 
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

### Pros
- **Automated Waste Sorting**: The system efficiently classifies waste into organic and recyclable categories, reducing manual sorting efforts.
  
- **Improved Efficiency**: Robust data preprocessing and feature extraction enhance the overall accuracy and speed of classification.
  
- **Real-Time Predictions**: The Flask API allows for immediate feedback on waste classification, facilitating better waste management decisions.
  
- **Environmental Impact**: Promotes recycling and sustainability by aiding in proper waste disposal practices.

### Challenges Faced

- **Memory Constraints**: Managing memory usage during image preprocessing was a significant challenge, especially with large datasets, requiring efficient data handling techniques.

- **Data Cleaning**: Ensuring robust data cleaning processes was essential to handle various image formats and remove corrupt files, which involved rigorous validation.

- **Integration Issues**: Integrating model predictions with a responsive web interface presented challenges in maintaining user experience while ensuring accurate and timely responses.

- **Misclassifications**: Overcoming occasional misclassifications required ongoing refinement of the model and careful evaluation of its performance on diverse input images.



---
✨ Developed by: Angel Kuria & Karanja Nyambura
