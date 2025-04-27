import os
import subprocess
import librosa
import numpy as np
import tensorflow as tf
import uuid

# ====== CONFIGURATION ======
VIDEO_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionFinal\test_video.mp4"  # Change this to your video path
TEMP_AUDIO_DIR = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionFinal\temp_audio"  # Temporary folder for storing extracted audio
MODEL_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionFinal\models\audio_deepfake_detector.h5"
FFMPEG_PATH = "ffmpeg"  # Ensure ffmpeg is installed and added to PATH

# Create temp directory if it doesn't exist
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# ====== Extract Audio from Single Video ======
def extract_audio(video_path):
    try:
        print("\n[Step 1] Extracting audio from video...")
        unique_filename = str(uuid.uuid4()) + ".wav"
        output_path = os.path.join(TEMP_AUDIO_DIR, unique_filename)

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
        print("Audio extraction complete!")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error during audio extraction: {e}")
        return None

# ====== Extract MFCC Features ======
def extract_mfcc(audio_path, max_pad_length=500):
    print("[Step 2] Extracting MFCC features...")
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)

    # Standardize
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    # Pad or truncate
    if mfcc.shape[1] < max_pad_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_length]

    print("Feature extraction complete!")
    return np.expand_dims(np.expand_dims(mfcc, axis=-1), axis=0)

# ====== Load Model ======
print("\n[Loading Model]...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# ====== Prediction Workflow ======
audio_path = extract_audio(VIDEO_PATH)

if audio_path:
    mfcc_features = extract_mfcc(audio_path)

    print("\n[Step 3] Predicting...")
    prediction = model.predict(mfcc_features, verbose=0)

    real_score = prediction[0][0]
    fake_score = prediction[0][1]
    predicted_class = np.argmax(prediction)

    label = "Real" if predicted_class == 0 else "Fake"

    print(f"\nPrediction Complete!")
    print(f"Real Score: {real_score:.4f}")
    print(f"Fake Score: {fake_score:.4f}")
    print(f"Final Prediction: {label} Audio")

    # Optional: Clean up temporary audio file
    os.remove(audio_path)
else:
    print("Prediction aborted due to audio extraction failure.")