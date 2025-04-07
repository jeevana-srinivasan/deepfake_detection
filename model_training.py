import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Paths
REAL_AUDIO_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionProject\audio\real"
FAKE_AUDIO_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionProject\audio\fake"

# MFCC Extraction
def extract_mfcc(audio_path, max_pad_length=500):
    print(f"Extracting MFCC from: {audio_path}")
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)

    # Normalize MFCC
    scaler = StandardScaler()
    mfcc = scaler.fit_transform(mfcc)

    # Pad or truncate
    if mfcc.shape[1] < max_pad_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_length]

    return np.expand_dims(mfcc, axis=-1)  # Add channel dimension

# Load dataset
X, y = [], []

# Load real audios
for file in os.listdir(REAL_AUDIO_PATH):
    if file.endswith(".wav"):
        X.append(extract_mfcc(os.path.join(REAL_AUDIO_PATH, file)))
        y.append(0)  # Label 0 for Real

# Load fake audios
for file in os.listdir(FAKE_AUDIO_PATH):
    if file.endswith(".wav"):
        X.append(extract_mfcc(os.path.join(FAKE_AUDIO_PATH, file)))
        y.append(1)  # Label 1 for Fake

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# One-hot encode the labels
y = to_categorical(y, num_classes=2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(2), y=np.argmax(y_train, axis=1))
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Define CNN Model
def create_model():
    model = Sequential([
        Input(shape=(100, 500, 1)),  # Input layer

        # First CNN Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Second CNN Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(256, activation='relu'),
        Dropout(0.5),

        Dense(2, activation='softmax')  # <- Two-class output
    ])
    return model

# Create and compile the model
model = create_model()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',   # <- Categorical crossentropy
              metrics=['accuracy'])

# Model Summary
model.summary()

# Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=3,            # <-- You can reduce if needed
                    batch_size=16,
                    class_weight=class_weight_dict)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")

# Save model
model.save("audio_deepfake_detector.h5")
print("Model saved successfully!")