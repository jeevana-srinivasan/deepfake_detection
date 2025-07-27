import os
import torch
import librosa
import numpy as np
import tensorflow as tf
import subprocess
from facenet_pytorch import MTCNN
from efficientnet_pytorch import EfficientNet
from PIL import Image
import cv2

# Paths
VIDEO_MODEL_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionFinal\models\efficientnet_faces.pth"
AUDIO_MODEL_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionFinal\models\audio_deepfake_detector.h5"
FFMPEG_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\ffmpeg\ffmpeg\bin\ffmpeg.exe"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models once
mtcnn = MTCNN(image_size=224, margin=0, device=DEVICE)

video_model = EfficientNet.from_pretrained('efficientnet-b0')
video_model._fc = torch.nn.Linear(video_model._fc.in_features, 2)
video_model.load_state_dict(torch.load(VIDEO_MODEL_PATH, map_location=DEVICE))
video_model.to(DEVICE)
video_model.eval()

audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)

# --- Video processing ---
def extract_faces_from_video(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    faces = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = mtcnn(img)
        if face is not None:
            faces.append(face)
    cap.release()

    if len(faces) < num_frames:
        pad_count = num_frames - len(faces)
        pad_tensor = torch.zeros((pad_count, 3, 224, 224), device=DEVICE)
        faces += [pad_tensor[i] for i in range(pad_count)]

    return torch.stack(faces) if faces else None

# --- Audio processing ---
def extract_audio_from_video(video_path, output_audio_path="temp_audio.wav"):
    try:
        subprocess.run([
            FFMPEG_PATH, "-y",
            "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            output_audio_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from {video_path}: {e}")

def extract_mfcc(audio_path, max_pad_length=500):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_length - mfcc.shape[1])), mode='constant') if mfcc.shape[1] < max_pad_length else mfcc[:, :max_pad_length]
    return np.expand_dims(mfcc, axis=(0, -1))

# --- Prediction Function for Web App ---
def predict_fused_model(video_path):
    faces = extract_faces_from_video(video_path)
    if faces is None:
        return {"error": "Face extraction failed."}

    # Video prediction
    faces = faces.to(DEVICE)
    with torch.no_grad():
        outputs = video_model(faces)
        video_scores = torch.softmax(outputs, dim=1)
        video_score = torch.mean(video_scores, dim=0)
        video_real_score = video_score[0].item()
        video_fake_score = video_score[1].item()

    # Audio prediction
    extract_audio_from_video(video_path)
    mfcc = extract_mfcc("temp_audio.wav")
    audio_preds = audio_model.predict(mfcc)
    audio_real_score = audio_preds[0][0]
    audio_fake_score = audio_preds[0][1]
    if os.path.exists("temp_audio.wav"):
        os.remove("temp_audio.wav")

    # Fusion logic
    #if video_fake_score > video_real_score or audio_fake_score > audio_real_score:
    #    predicted_label = "fake"
    ##else:
    #    predicted_label = "real"
    # Determine which modality predicted fake
    video_is_fake = video_fake_score > video_real_score
    audio_is_fake = audio_fake_score > audio_real_score

    if video_is_fake or audio_is_fake:
        predicted_label = "fake"
        reasons = []
        if video_is_fake:
            reasons.append("video")
        if audio_is_fake:
            reasons.append("audio")
        explanation = "Fake detected based on: " + " and ".join(reasons).capitalize()
    else:
        predicted_label = "real"
        explanation = "Both audio and video were predicted as real."

    return {
        "predicted_label": predicted_label,
        "video_real_score": video_real_score,
        "video_fake_score": video_fake_score,
        "audio_real_score": audio_real_score,
        "audio_fake_score": audio_fake_score,
        "explanation": explanation
    }