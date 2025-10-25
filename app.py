import os
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------
# 🌿 Flask App Configuration
# ---------------------------------------------------------------
app = Flask(__name__)

# Directory to store uploaded images temporarily
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model configuration
MODEL_PATH = 'plant_health_model.keras'
IMAGE_SIZE = (224, 224)

# ---------------------------------------------------------------
# 🧠 Load the trained model
# ---------------------------------------------------------------
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ ERROR: Could not load model from {MODEL_PATH}. Details: {e}")
    model = None

# ---------------------------------------------------------------
# 🌿 Prediction Helper Function
# ---------------------------------------------------------------
def model_predict(file_path):
    """Run prediction on the uploaded image."""
    if model is None:
        return {'prediction': 'Model not loaded', 'confidence': 0.0, 'class_id': -1}

    try:
        # Load and preprocess the image
        img = image.load_img(file_path, target_size=IMAGE_SIZE)
        img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

        # Predict
        prediction_raw = model.predict(img_array)[0][0]
        class_id = 1 if prediction_raw > 0.5 else 0
        confidence = prediction_raw if class_id == 1 else 1 - prediction_raw

        return {
            'prediction': "Diseased Leaf 🦠" if class_id == 1 else "Healthy Leaf 🌱",
            'confidence': round(confidence * 100, 2),
            'class_id': class_id
        }

    except Exception as e:
        print(f"⚠️ Prediction Error: {e}")
        return {'prediction': f'Error: {e}', 'confidence': 0.0, 'class_id': -1}

# ---------------------------------------------------------------
# 🏠 Home Route — Upload Page
# ---------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

# ---------------------------------------------------------------
# 🔍 Predict Route — Handles Upload and Model Inference
# ---------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save uploaded image
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"📸 Uploaded file saved to: {file_path}")

        # Run prediction
        results = model_predict(file_path)

        # ✅ Correct image URL path
        image_url = f"/uploads/{filename}"

        # Render the result page
        return render_template(
            'result.html',
            prediction=results['prediction'],
            confidence=f"{results['confidence']:.2f}%",
            is_healthy=(results['class_id'] == 0),
            image_url=image_url
        )

# ---------------------------------------------------------------
# 🖼️ Route to Serve Uploaded Files
# ---------------------------------------------------------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves the uploaded image for display on the result page."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ---------------------------------------------------------------
# 🚀 Run Flask App
# ---------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
