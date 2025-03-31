from flask import Flask, render_template, request, send_from_directory
import numpy as np
import os
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ✅ Load the trained model
model_path = os.path.join(os.getcwd(), "svm_model.pkl")


try:
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Define model even if loading fails

# ✅ Define image preprocessing function
def preprocess_image(filepath):
    """Loads and preprocesses the image for model inference."""
    image = Image.open(filepath).convert('RGB')  # ✅ Convert to RGB
    image = np.array(image, dtype=np.float32) / 255.0  # ✅ Normalize to [0,1]
    
    # ✅ Resize using TensorFlow (matches training pipeline)
    image = tf.image.resize(image, (128, 128))  
    
    # ✅ Ensure correct shape for model
    image = np.expand_dims(image, axis=0)  # Shape: (1, 128, 128, 3)
    
    return image

# 1️⃣ Home Page
@app.route('/')
def home():
    return render_template('index.html')

# 2️⃣ About Page
@app.route('/about')
def about():
    return render_template('about.html')

# 3️⃣ Contact Page
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/classify')
def classify():
    return render_template('classify.html')

# ✅ Prediction Route
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
        print(f"📸 Image saved at: {filepath}")

        # ✅ Preprocess image
        image_array = preprocess_image(filepath)

        # ✅ Make prediction
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]  
        class_names = ['Organic', 'Recyclable']
        predicted_class = class_names[predicted_class]

        return render_template('classify.html', prediction=predicted_class, image_url=f'/uploads/{file.filename}')
    except Exception as e:
        return render_template('classify.html', prediction=f"Error: {str(e)}")

# ✅ Serve Uploaded Images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
