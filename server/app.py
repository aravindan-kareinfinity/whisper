from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import os

app = Flask(__name__)
CORS(app)

# Load the Whisper model
model = whisper.load_model("base")

@app.route('/process-audio', methods=['POST'])
def process_audio():
    try:
        # Get the audio file from the request
        audio_file = request.files['audio']
        
        # Save the audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_file.save(temp_audio.name)
            
            # Transcribe the audio
            result = model.transcribe(temp_audio.name)
            
            # Clean up the temporary file
            os.unlink(temp_audio.name)
            
            return jsonify({
                'success': True,
                'text': result['text']
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 