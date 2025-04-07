import os
import random
import torch
import librosa
import numpy as np
import tensorflow as tf
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
import cv2
import subprocess

# Paths
TEST_DATA_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionFinal\Test_videos"
VIDEO_MODEL_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionFinal\models\efficientnet_model.pth"
AUDIO_MODEL_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionFinal\models\audio_deepfake_detector.h5"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Models
def load_video_model():
    model = EfficientNet.from_name("efficientnet-b0", num_classes=2)
    model.load_state_dict(torch.load(VIDEO_MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

def load_audio_model():
    return tf.keras.models.load_model(AUDIO_MODEL_PATH)

# Preprocessing
video_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_video_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Warning: {video_path} has no frames!")
        return None
    selected_frames = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(video_transform(frame))
    cap.release()
    return torch.stack(frames) if frames else None

def extract_audio_from_video(video_path, output_audio_path="temp_audio.wav"):
    output_dir = os.path.dirname(output_audio_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_audio_path
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from {video_path}: {e}")

def extract_mfcc(audio_path, max_pad_length=500):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)

    # Standardize the MFCC
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    # Pad or truncate
    if mfcc.shape[1] < max_pad_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_length]

    # Expand dims (important for feeding into CNN)
    return np.expand_dims(mfcc, axis=-1)

# Prediction
def predict(video_path, video_model, audio_model):
    frames = extract_video_frames(video_path)
    if frames is None:
        return None

    frames = frames.to(DEVICE)

    # Predict for each frame individually and average
    video_outputs = []
    with torch.no_grad():
        for frame in frames:
            frame = frame.unsqueeze(0)  # Add batch dimension
            output = video_model(frame)
            video_outputs.append(output)

    video_outputs = torch.cat(video_outputs, dim=0)
    video_output = torch.mean(video_outputs, dim=0)  # Average over frames

    video_scores = torch.softmax(video_output, dim=0)
    video_real_score = video_scores[0].item()
    video_fake_score = video_scores[1].item()

    # Extract and predict audio
    extract_audio_from_video(video_path)

    mfcc = extract_mfcc('temp_audio.wav')    # Extract MFCC features
    mfcc = np.expand_dims(mfcc, axis=0)       # Add batch dimension like in audio_test.py

    audio_preds = audio_model.predict(mfcc)   # Predict
    audio_real_score = audio_preds[0][0]       # Real score
    audio_fake_score = audio_preds[0][1]       # Fake score

    # Clean up temp file
    if os.path.exists('temp_audio.wav'):
        os.remove('temp_audio.wav')

    predicted_label = 'real' if (0.7 * video_real_score + 0.3 * audio_real_score) > (0.7 * video_fake_score + 0.3 * audio_fake_score) else 'fake'

    return {
        "predicted_label": predicted_label,
        "audio_real_score": audio_real_score,
        "audio_fake_score": audio_fake_score,
        "video_real_score": video_real_score,
        "video_fake_score": video_fake_score
    }

def select_random_videos(test_data_path, num_videos=10):
    video_files = []
    for label in ["real", "fake"]:
        folder_path = os.path.join(test_data_path, label)
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith(".mp4"):
                    video_files.append((os.path.join(folder_path, f), label))
    return random.sample(video_files, num_videos)

def main():
    video_model = load_video_model()
    audio_model = load_audio_model()

    random_videos = select_random_videos(TEST_DATA_PATH, num_videos=20)

    correct = 0

    for idx, (video_path, actual_label) in enumerate(random_videos):
        print(f"\n====== Video {idx+1} ======")
        print(f"Actual label: {actual_label}")
        result = predict(video_path, video_model, audio_model)

        if result is None:
            print("Skipping video due to frame extraction error.")
            continue

        print(f"Model Prediction: {result['predicted_label']}")
        print(f"Audio - real_score: {result['audio_real_score']:.4f}    fake_score: {result['audio_fake_score']:.4f}")
        print(f"Video - real_score: {result['video_real_score']:.4f}    fake_score: {result['video_fake_score']:.4f}")

        if result['predicted_label'] == actual_label:
            correct += 1

    accuracy = (correct / 20) * 100
    print(f"Overall Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()