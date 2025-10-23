import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import uuid

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'plant_health_model.keras'
TARGET_SIZE = (96, 96)
LABELS = {0: 'Diseased', 1: 'Healthy'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Ensure uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Checks if file type is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def model_predict(img_path, model):
    """Predicts the health of the plant from the image."""
    if model is None:
        return "Model Error", 0.0

    img = image.load_img(img_path, target_size=TARGET_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = LABELS[1] if prediction >= 0.5 else LABELS[0]
    confidence = prediction * 100 if label == 'Healthy' else (1 - prediction) * 100

    return label, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    """Handles multiple image uploads."""
    if 'files[]' not in request.files:
        return redirect(url_for('index'))

    files = request.files.getlist('files[]')
    results = []

    for file in files:
        if file and allowed_file(file.filename):
            unique_filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            label, confidence = model_predict(filepath, model)

            results.append({
                'filename': unique_filename,
                'label': label,
                'confidence': f"{confidence:.2f}%"
            })

    return render_template('result.html', results=results)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
