import os
import random
import subprocess
import librosa
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Paths
TEST_VIDEO_DIR = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionFinal\Test_videos"
OUTPUT_AUDIO_DIR = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionFinal\Test_audios" #create a empty folder and specify the path here to save the audio files
MODEL_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionFinal\models\audio_deepfake_detector.h5"
FFMPEG_PATH = "ffmpeg"

# Ensure output directories exist
os.makedirs(os.path.join(OUTPUT_AUDIO_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_AUDIO_DIR, "fake"), exist_ok=True)

# Get unique filename
def get_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    unique_filename = filename
    while os.path.exists(os.path.join(directory, unique_filename)):
        unique_filename = f"{base}_{counter}{ext}"
        counter += 1
    return os.path.join(directory, unique_filename)

# Extract audio from videos
def extract_audio_from_videos(video_folder, category):
    extracted_files = []
    output_dir = os.path.join(OUTPUT_AUDIO_DIR, category)

    for root, _, files in os.walk(video_folder):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                output_filename = os.path.splitext(file)[0] + ".wav"
                output_path = get_unique_filename(output_dir, output_filename)

                try:
                    ffmpeg_cmd = [
                        FFMPEG_PATH,
                        "-i", video_path,
                        "-vn",
                        "-acodec", "pcm_s16le",
                        "-ar", "16000",
                        "-ac", "1",
                        output_path
                    ]
                    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    extracted_files.append((output_path, category))
                except subprocess.CalledProcessError as e:
                    print(f"Error extracting audio from {video_path}: {e}")
    return extracted_files

# Extract MFCCs
def extract_mfcc(audio_path, max_pad_length=500):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)

    # Standardize
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    # Pad or truncate
    if mfcc.shape[1] < max_pad_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_length]

    return np.expand_dims(mfcc, axis=-1)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Extract audios
real_videos_path = os.path.join(TEST_VIDEO_DIR, "real")
fake_videos_path = os.path.join(TEST_VIDEO_DIR, "fake")

test_files = extract_audio_from_videos(real_videos_path, "real")
test_files += extract_audio_from_videos(fake_videos_path, "fake")

# Testing
actual_labels = []
predicted_labels = []

label_mapping = {"real": 0, "fake": 1}
reverse_mapping = {0: "Real", 1: "Fake"}

for audio_path, category in test_files:
    mfcc = extract_mfcc(audio_path)
    mfcc = np.expand_dims(mfcc, axis=0)  # Make batch dimension

    prediction = model.predict(mfcc)
    real_score = prediction[0][0]
    fake_score = prediction[0][1]
    predicted_class = np.argmax(prediction)

    actual_label = label_mapping[category]

    actual_labels.append(actual_label)
    predicted_labels.append(predicted_class)

    print(f"File: {os.path.basename(audio_path)}")
    print(f"Actual Label: {reverse_mapping[actual_label]}")
    print(f"Predicted Label: {reverse_mapping[predicted_class]}")
    print(f"Real Score: {real_score:.3f}")
    print(f"Fake Score: {fake_score:.3f}")
    print("-" * 40)

# Overall accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(actual_labels, predicted_labels)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
report = classification_report(actual_labels, predicted_labels, target_names=["Real", "Fake"])
print("\nClassification Report:")
print(report)