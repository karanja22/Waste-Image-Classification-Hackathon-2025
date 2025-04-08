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
## Feature Extraction 

### Overview

In our project, we performed feature extraction to transform raw image data into a format suitable for machine learning algorithms.

 1. **Loading Images**: We began by loading the preprocessed images from storage, allowing us to manipulate and analyze them effectively.
 2. **Reshaping Images**: Each image, originally a 2D array (height x width) with color channels, was reshaped into a 1D array (feature vector). For instance, a 256x256 image with RGB channels was converted into a vector of size 256 * 256 *
 3. **Compiling Feature Set**: We compiled all the reshaped images into a single array, forming the feature set X. This array served as the input for training our model.

### Code

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load images and labels
X = np.load("images.npy").reshape(-1, 256 * 256 * 3)  # Reshape images
y = np.load("labels.npy")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## Data splitting

### Overview

Rather than loading the entire dataset into memory (which can cause memory issues), we processed and saved the data in batches using NumPy, then used scikit-learn’s train_test_split to generate indices for training and testing, ensuring a stratified split.
 1. **Training Set**: This subset is used to train the model. It contains the majority of the data (e.g., 80%) and allows the model to learn the underlying patterns in the data.
 2. **Testing Set**: The remaining subset (e.g., 20%) is used to evaluate the model's performance. This helps assess how well the model generalizes to unseen data.
 3. **Stratification**: When splitting the data, it's important to maintain the proportion of each class (e.g., organic vs. recyclable waste) in both subsets. This ensures that the model is trained and tested on a representative sample of each class.
    
### Code 

```python
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
train_indices, test_indices = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
```


---
## Model Training

###Overview

Model training is the process of teaching the machine learning algorithm to recognize patterns in the data.

1. **Choosing a Model**: For this project, we use SGDClassifier, which is a linear classifier that uses stochastic gradient descent for optimization. This model is effective for large datasets and can handle online learning.
2. **Fitting the Model**: The model is trained using the training set (X_train, y_train). During this process, the algorithm adjusts its internal parameters to minimize the error in predictions based on the training data.
3. **Making Predictions**: After training, the model is tested on the testing set (X_test). It predicts the class labels for the test images.
4. **Evaluating Performance**: Finally, the model's performance is evaluated using metrics such as accuracy, which indicates the proportion of correct predictions compared to the total number of predictions. This helps determine how well the model will perform in real-world scenarios.

### Code

```python
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# Initialize and train the model
model = SGDClassifier(loss='log_loss', random_state=42)
model.fit(X_train, y_train)

# Evaluate performance
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model accuracy: {accuracy:.2f}")
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


---
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

---
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

![classify](https://github.com/user-attachments/assets/b2834ec6-d1db-4f90-9e37-0a5b99b68283)


##### About page

![about (2)](https://github.com/user-attachments/assets/a6b4c5d3-3128-438d-8d25-fc7d16aad1a7)

##### Contact page

![contact](https://github.com/user-attachments/assets/9243b397-0220-4fe1-8167-0c50c8255e5a)


## Classified Images from our model

![pred](https://github.com/user-attachments/assets/f6d5dfda-f615-4d56-8e56-be259de84822)


![pred2](https://github.com/user-attachments/assets/12811173-aa85-4ba1-85b1-9d7e32d472f0)


---
## Conclusion

### Achievements

- **Automated Waste Sorting**: The system efficiently classifies waste into organic and recyclable categories, reducing manual sorting efforts.
  
- **Improved Efficiency**: Robust data preprocessing and feature extraction enhance the overall accuracy and speed of classification.
  
- **Real-Time Predictions**: The Flask API allows for immediate feedback on waste classification, facilitating better waste management decisions.
  
- **Environmental Impact**: Promotes recycling and sustainability by aiding in proper waste disposal practices.

### Challenges Faced

- **Memory Constraints**: Managing memory usage during image preprocessing was a significant challenge, especially with large datasets, requiring efficient data handling techniques.

- **Data Cleaning**: Ensuring robust data cleaning processes was essential to handling various image formats and removing corrupt files, which involved rigorous validation.

- **Integration Issues**: Integrating model predictions with a responsive web interface presented challenges in maintaining user experience while ensuring accurate and timely responses.

- **Misclassifications**: Overcoming occasional misclassifications required ongoing refinement of the model and careful evaluation of its performance on diverse input images.



---
✨ Developed by: Angel Kuria & Karanja Nyambura
