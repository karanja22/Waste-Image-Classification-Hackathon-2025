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
   All images were resized to **128×128 pixels** to ensure consistency across the dataset.

3. **Normalization:**  
   Pixel values were normalized to the range **[0, 1]** to facilitate efficient training.

### Code Snippet

### Challenges Faced

- **Memory Constraints:**  
  Handling large datasets required batch processing using the `tf.data.Dataset` API.

- **Data Inconsistencies:**  
  Some images were corrupt or in different formats, necessitating robust error handling in the preprocessing code.

---
## Feature Extraction

### Overview

Feature extraction transforms raw image data into a structured format that a machine learning model can understand.

Unlike the initial approach where we considered grayscale images, we trained our model using full RGB images to retain color-based features that improve classification accuracy.

### Feature Representation

Each 128×128 RGB image consists of three color channels (Red, Green, Blue), resulting in a shape of (128, 128, 3). We normalized these pixel values to [0,1] before feeding them into the model.
---
## Data splitting

### Overview

After feature extraction, we split the dataset into training (80%) and validation (20%) sets to train and evaluate our model's performance. Instead of loading the entire dataset into memory at once (which can cause memory overload), we implemented an efficient data pipeline using TensorFlow’s tf.data API to load, preprocess, and feed data to the model dynamically in batches.

### Code Snippet for feature extraction and data splitiing
---

## Model Training



## Model Evaluation


## Deployment & API

### Overview

To integrate our trained waste classification model into a real-world application, we developed a Flask API that serves predictions in real time. This API allows users to upload an image of waste, which is then processed and classified into one of the predefined categories. Additionally, the API provides useful waste handling tips based on the classification result.

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
- **Model predicting wrongly.

This project not only highlights technical expertise in machine learning and web development but also underscores the potential for AI-driven solutions to contribute to a cleaner, more sustainable environment.


---
✨ Developed by: Angel Kuria & Karanja Nyambura
