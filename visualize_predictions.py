# visualize_predictions.py
# 🍃 Display sample classified images from the test dataset

import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('plant_health_model.h5')

# Define the path to your test dataset
test_dir = 'dataset/test'

# Get folder paths
categories = ['healthy', 'diseased']

# Pick random samples from both categories
plt.figure(figsize=(10, 6))

for i in range(6):
    # Pick a random category and random image
    category = random.choice(categories)
    folder = os.path.join(test_dir, category)
    file = random.choice(os.listdir(folder))
    img_path = os.path.join(folder, file)
    
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "🌿 Diseased" if prediction > 0.5 else "🍃 Healthy"
    
    # Display
    plt.subplot(2, 3, i + 1)
    plt.imshow(image.load_img(img_path))
    plt.title(f"Predicted: {label}")
    plt.axis('off')

plt.tight_layout()
plt.show()
