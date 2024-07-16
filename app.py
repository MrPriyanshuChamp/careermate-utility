from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os
import tempfile

# Initialize the Flask app
app = Flask(__name__)
CORS(app, resources={r"/transcribe": {"origins": ["http://localhost:3000","https://careermate.vercel.app"]}})
# Load the Whisper model
model = whisper.load_model('base')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        try:
            # Transcribe the audio file
            result = model.transcribe(temp_file_path)
        finally:
            # Remove the temporary file
            os.remove(temp_file_path)

        # Return the result as JSON
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
