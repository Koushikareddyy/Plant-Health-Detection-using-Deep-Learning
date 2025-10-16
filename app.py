import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'plant_health_model.h5' 
TARGET_SIZE = (128, 128) 
# Assumes 'Diseased' is class 0 and 'Healthy' is class 1 (check your generator output)
LABELS = {0: 'Diseased', 1: 'Healthy'} 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the model once when the app starts
try:
    # Disable eager execution for better performance with Flask
    tf.compat.v1.disable_eager_execution() 
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def model_predict(img_path, model):
    """Preprocesses the image and makes a prediction."""
    if model is None:
        return "Model Error", 0.0

    img = image.load_img(img_path, target_size=TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0 # Normalize 
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    # Predict (output is probability of class 1: Healthy)
    prediction = model.predict(img_array)[0]
    
    prob_healthy = prediction[0]
    
    if prob_healthy >= 0.5:
        label = LABELS[1]
        confidence = prob_healthy * 100
    else:
        label = LABELS[0]
        confidence = (1 - prob_healthy) * 100 # Confidence in the Diseased class
    
    return label, confidence

@app.route('/', methods=['GET'])
def index():
    """Renders the initial upload form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    """Handles image upload and prediction."""
    if 'file' not in request.files:
        return redirect(url_for('index'))
        
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('index'))
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make Prediction
        label, confidence = model_predict(filepath, model)
        
        # Clean up the file
        os.remove(filepath)
        
        # Display result
        return render_template('result.html', label=label, confidence=f"{confidence:.2f}%")

if __name__ == '__main__':
    # Run the application
    app.run(debug=True, port=5000)