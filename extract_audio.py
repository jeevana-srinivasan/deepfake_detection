import os
import subprocess

# FIXED: Use raw strings to avoid escape sequence errors
BASE_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionFinal\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"

# FIXED: Use raw strings for paths
OUTPUT_DIR = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\AudioExtractionFinal\FakeAVCeleb_audio"
os.makedirs(os.path.join(OUTPUT_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "fake"), exist_ok=True)

# FIXED: Correct category mapping
CATEGORY_MAPPING = {
    "RealVideo-RealAudio": "real",  
    "RealVideo-FakeAudio": "fake",
}

# FIXED: Set explicit FFmpeg path
FFMPEG_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\ffmpeg\ffmpeg\bin\ffmpeg.exe"
if not os.path.exists(FFMPEG_PATH):
    raise FileNotFoundError(f"FFmpeg not found at: {FFMPEG_PATH}. Check the path.")

# Function to generate a unique filename if file exists
def get_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename

    while os.path.exists(os.path.join(directory, new_filename)):  # Check if file exists
        new_filename = f"{base}_{counter}{ext}"  # Append counter
        counter += 1

    return os.path.join(directory, new_filename)

# Function to extract audio from video
def process_videos(video_folder, category):
    for root, _, files in os.walk(video_folder):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                output_dir = os.path.join(OUTPUT_DIR, category)
                
                output_filename = os.path.splitext(file)[0] + ".wav"
                output_path = get_unique_filename(output_dir, output_filename)  # Ensure unique name

                try:
                    # FIXED: Pass command as a list (handles spaces & special characters)
                    ffmpeg_cmd = [
                        FFMPEG_PATH, "-i", video_path, "-vn",
                        "-acodec", "pcm_s16le", "-ar", "16000", output_path
                    ]
                    subprocess.run(ffmpeg_cmd, check=True)
                    print(f"Extracted: {output_path}")

                except subprocess.CalledProcessError as e:
                    print(f"Error processing {video_path}: {str(e)}")

# Run for all categories
for folder, category in CATEGORY_MAPPING.items():
    folder_path = os.path.join(BASE_PATH, folder)
    if os.path.exists(folder_path):  # Ensure the folder exists before processing
        process_videos(folder_path, category)
    else:
        print(f"Warning: Folder {folder_path} does not exist. Skipping...")

print("AUDIO EXTRACTION COMPLETED!")