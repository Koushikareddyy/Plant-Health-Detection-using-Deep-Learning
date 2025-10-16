# app.py 🌿 Plant Health Detection Web App

import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'plant_health_model.keras'   # ✅ use your latest model file
TARGET_SIZE = (96, 96)                    # ✅ matches your model input size
LABELS = {0: 'Diseased 🍂', 1: 'Healthy 🌿'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load the trained model once ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- Helper Functions ---
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def model_predict(img_path, model):
    """Preprocess the image and return prediction + confidence."""
    if model is None:
        return "Model not loaded", 0.0

    img = image.load_img(img_path, target_size=TARGET_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]  # sigmoid output → single probability

    if prediction > 0.5:
        label = LABELS[1]
        confidence = prediction * 100
    else:
        label = LABELS[0]
        confidence = (1 - prediction) * 100

    return label, confidence

# --- Routes ---
@app.route('/')
def index():
    """Home page with upload form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and make prediction."""
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    label, confidence = model_predict(filepath, model)

    # keep image so result.html can display it
    return render_template(
        'result.html',
        label=label,
        confidence=f"{confidence:.2f}%",
        image_url=url_for('uploaded_file', filename=filename)
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files for display."""
    return tf.io.gfile.GFile(os.path.join(UPLOAD_FOLDER, filename), 'rb').read()

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
