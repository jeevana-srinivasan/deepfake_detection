from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__, template_folder='templates')  # <-- tell Flask where templates are!

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')  # <-- THIS SERVES YOUR FRONTEND!

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # your model prediction logic here...
        prediction = "FAKE"  # or "FAKE"

        # IMPORTANT: always return jsonify
        return jsonify({'prediction': prediction})
    
# ADD THIS PART:
if __name__ == '__main__':
    app.run(debug=True)
