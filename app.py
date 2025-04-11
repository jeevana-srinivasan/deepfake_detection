from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import datetime
import csv
import subprocess

# Import prediction function
from fused_predict import predict_fused_model

app = Flask(__name__, template_folder='templates', static_folder='static')

FFMPEG_PATH = r"C:\Users\srins\Documents\JeevanaSrinivasan\8th sem\major_project\ffmpeg\ffmpeg\bin\ffmpeg.exe"

# Upload folder
UPLOAD_FOLDER = 'uploads'
REENCODED_FOLDER = 'reencoded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REENCODED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Log file
LOG_DIR = 'logs'
LOG_FILE = os.path.join(LOG_DIR, 'prediction_log.csv')
os.makedirs(LOG_DIR, exist_ok=True)

# Logging function
def log_prediction(filename, result):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = [
        timestamp,
        filename,
        result['predicted_label'],
        round(result['video_real_score'], 4),
        round(result['video_fake_score'], 4),
        round(result['audio_real_score'], 4),
        round(result['audio_fake_score'], 4)
    ]
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Filename", "Prediction",
                             "Video_Real", "Video_Fake", "Audio_Real", "Audio_Fake"])
        writer.writerow(row)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_and_reencode', methods=['POST'])
def upload_and_reencode():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded.'})
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    # Re-encode
    output_filename = 'reencoded_' + filename
    output_path = os.path.join(REENCODED_FOLDER, output_filename)

    command = [
        'ffmpeg', '-i', input_path,
        '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac',
        output_path,
        '-y'  # Overwrite
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        reencoded_url = '/reencoded/' + output_filename
        return jsonify({'success': True, 'reencoded_url': reencoded_url})
    except subprocess.CalledProcessError as e:
        return jsonify({'success': False, 'error': str(e)})

# Route to serve re-encoded videos
@app.route('/reencoded/<path:filename>')
def serve_reencoded(filename):
    return send_from_directory(REENCODED_FOLDER, filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        result = predict_fused_model(video_path)
        log_prediction(filename, result)

        if os.path.exists(video_path):
            os.remove(video_path)

        if "error" in result:
            return jsonify({'error': result['error']}), 500
        
        reencoded_folder = 'reencoded'
        try:
            for filename in os.listdir(reencoded_folder):
                file_path = os.path.join(reencoded_folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            print(f"Error clearing reencoded files: {e}")

        #return jsonify({'prediction': result['predicted_label'].upper()})
        return jsonify({
            'prediction': result['predicted_label'].upper(),
            'explanation': result['explanation']
        })


@app.route('/logs')
def view_logs():
    if not os.path.exists(LOG_FILE):
        return "No logs available yet."
    with open(LOG_FILE, 'r') as f:
        log_data = f.read().replace('\n', '<br>')
    return f"<div style='padding: 20px; font-family: monospace;'>{log_data}</div>"

if __name__ == '__main__':
    app.run(debug=True)
