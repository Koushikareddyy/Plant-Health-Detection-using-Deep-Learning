# plant_health.py
# 🌿 Plant Disease Detection using Transfer Learning (MobileNetV2 + Keras)

import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------------
# 1️⃣ Dataset Paths
# ---------------------------------------------------------------
train_dir = 'dataset/train'
test_dir = 'dataset/test'

print("📂 Checking dataset structure...")
try:
    print("Training folders:", os.listdir(train_dir))
    print("Testing folders:", os.listdir(test_dir))
except FileNotFoundError:
    print("❌ Dataset not found. Ensure 'dataset/train' and 'dataset/test' exist.")
    exit()

# ---------------------------------------------------------------
# 2️⃣ Data Preprocessing
# ---------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(96, 96),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(96, 96),
    batch_size=32,
    class_mode='binary'
)

# ---------------------------------------------------------------
# 3️⃣ Model Definition (MobileNetV2 Transfer Learning)
# ---------------------------------------------------------------
print("🧠 Building MobileNetV2 Transfer Learning Model...")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
base_model.trainable = False  # freeze base layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# ---------------------------------------------------------------
# 4️⃣ Train the Model (Base Training)
# ---------------------------------------------------------------
print("🚀 Starting training...")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    callbacks=[early_stopping]
)

# ---------------------------------------------------------------
# 5️⃣ Fine-tune (Unfreeze last layers)
# ---------------------------------------------------------------
print("🔧 Fine-tuning last layers...")
base_model.trainable = True
for layer in base_model.layers[:-30]:  # freeze all but last 30 layers
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

fine_tune_history = model.fit(
    train_generator,
    epochs=5,
    validation_data=test_generator
)

# ---------------------------------------------------------------
# 6️⃣ Save the Model
# ---------------------------------------------------------------
model.save('plant_health_model.keras')
print("✅ Model saved as 'plant_health_model.keras'.")

# ---------------------------------------------------------------
# 7️⃣ Evaluate & Plot
# ---------------------------------------------------------------
loss, accuracy = model.evaluate(test_generator)
print(f"📊 Final Test Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'] + fine_tune_history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('📈 Model Accuracy (with Fine-Tuning)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

