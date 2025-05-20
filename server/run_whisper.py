import whisper

# Load the medium model
print("Loading Whisper model...")
model = whisper.load_model("medium")

# Transcribe audio with language specification
print("\nTranscribing audio...")
result = model.transcribe("test_audio.mp3", language="en")

# Print the transcription
print("\nTranscription:")
print(result["text"]) 