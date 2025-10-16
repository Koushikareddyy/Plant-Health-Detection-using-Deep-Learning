# plant_health.py
# 🌿 Plant Disease Detection using CNN (TensorFlow + Keras)

import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------------
# 1️⃣ Paths to dataset folders
# ---------------------------------------------------------------
train_dir = 'dataset/train'
test_dir = 'dataset/test'

print("📂 Checking dataset structure...")
try:
    print("Training folders:", os.listdir(train_dir))
    print("Testing folders:", os.listdir(test_dir))
except FileNotFoundError:
    print("Error: Dataset folders not found. Please ensure 'dataset/train' and 'dataset/test' exist.")

# ---------------------------------------------------------------
# 2️⃣ Data Preprocessing (Rescale & Augment)
# ---------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,           
    shear_range=0.2,          
    zoom_range=0.2,           
    horizontal_flip=True,     
    rotation_range=20,        
    fill_mode='nearest'       
)

test_datagen = ImageDataGenerator(rescale=1./255) 

# CRITICAL FIXES for training stalls
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),   
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    workers=1,                 
    use_multiprocessing=False  
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    workers=1,
    use_multiprocessing=False
)

# ---------------------------------------------------------------
# 3️⃣ Define the CNN Model (FIXED FOR OVERFITTING)
# ---------------------------------------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Dropout(0.5), 
    
    Flatten(),
    
    Dense(64, activation='relu'), 
    
    Dropout(0.5), 
    
    Dense(1, activation='sigmoid')
])

# ---------------------------------------------------------------
# 4️⃣ Compile the Model
# ---------------------------------------------------------------
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ---------------------------------------------------------------
# 5️⃣ Train the Model (ADDED EARLY STOPPING)
# ---------------------------------------------------------------
early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=5,             
    restore_best_weights=True 
)

print("🚀 Starting training...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50, 
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=[early_stopping] 
)

# ---------------------------------------------------------------
# 6️⃣ Save the Trained Model
# ---------------------------------------------------------------
model.save('plant_health_model.h5')
print("✅ Training complete! Model saved as 'plant_health_model.h5'.")

# ---------------------------------------------------------------
# 7️⃣ Evaluate Model Performance & Plot
# ---------------------------------------------------------------
loss, accuracy = model.evaluate(test_generator)
print(f"📊 Final Test Accuracy: {accuracy * 100:.2f}%")

# Plot training accuracy and loss
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()