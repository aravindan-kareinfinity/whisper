from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import whisper
import numpy as np
import asyncio
import json
import base64
import wave
import io
import tempfile
import os

app = FastAPI(
    title="Whisper Real-time Transcription API",
    description="WebSocket API for real-time audio transcription using OpenAI's Whisper model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model (you can change the model size as needed)
model = whisper.load_model("base")

class AudioBuffer:
    def __init__(self):
        self.buffer = []
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        self.channels = 1  # Whisper expects mono audio

    def add_audio(self, audio_data):
        self.buffer.extend(audio_data)

    def get_audio(self):
        return np.array(self.buffer)

    def clear(self):
        self.buffer = []

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            print(f"Received message: {message['type']}")
            
            if message["type"] == "audio":
                audio_bytes = base64.b64decode(message["data"])

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_bytes)
                    temp_file_path = temp_file.name

                result = model.transcribe(temp_file_path)

                await websocket.send_json({
                    "type": "transcription",
                    "text": result["text"]
                })

                os.unlink(temp_file_path)

            elif message["type"] == "end":
                break

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {str(e)}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 