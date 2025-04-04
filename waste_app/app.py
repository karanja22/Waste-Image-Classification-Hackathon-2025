from flask import Flask, render_template, request, send_from_directory
import numpy as np
import os
from PIL import Image
import joblib
from flask_cors import CORS


app = Flask(__name__)

CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model_path = "svm_model.pkl"
scaler_path = "scaler.pkl"


try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Define model even if loading fails

try:
    scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    scaler = None  # Define model even if loading fails

# Define image preprocessing function
def preprocess_image(filepath):
    """Loads and preprocesses the image for model inference."""
    image = Image.open(filepath).convert('RGB')  # Convert to RGB
    image = image.resize((256, 256))  # Resize the image to match model input
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize the image to [0,1]
    image = image.flatten()  # Flatten the image into a 1D array (height * width * channels)
    
    print(f"Shape before expanding dimensions: {image.shape}")  # Debugging: Check the shape

    # Expand dimensions to add the sample axis (1, number of features)
    image = np.expand_dims(image, axis=0)  # Shape: (1, 196608)
    
    print(f"Processed image shape: {image.shape}")  # Debugging: Check the shape after expansion
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

        # Preprocess image
        image_array = preprocess_image(filepath)

        # Apply the scaler to the image to match the model's scale
        image_array = scaler.transform(image_array)

        # Make prediction
        predictions = model.predict(image_array)

        # If the model is binary, we may not need to use np.argmax
        # For binary classification, predictions should directly give the class
        class_names = ['Organic', 'Recyclable']
        
        predicted_class = class_names[int(predictions[0])]  # Using predictions directly for binary classification

        return render_template('classify.html', prediction=predicted_class, image_url=f'/uploads/{file.filename}')
    except Exception as e:
        return render_template('classify.html', prediction=f"Error: {str(e)}")


# Serve Uploaded Images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
