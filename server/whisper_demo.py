import whisper
import argparse

def transcribe_audio(audio_path, model_name="base", language=None, translate=False):
    """
    Transcribe audio using Whisper
    
    Args:
        audio_path (str): Path to the audio file
        model_name (str): Model size (tiny, base, small, medium, large)
        language (str): Language code (e.g., 'en' for English)
        translate (bool): Whether to translate to English
    """
    print(f"Loading {model_name} model...")
    model = whisper.load_model(model_name)
    
    print("Transcribing audio...")
    result = model.transcribe(
        audio_path,
        language=language,
        task="translate" if translate else "transcribe"
    )
    
    print("\nTranscription:")
    print(result["text"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper")
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                      help="Model size to use")
    parser.add_argument("--language", help="Language code (e.g., 'en' for English)")
    parser.add_argument("--translate", action="store_true", help="Translate to English")
    
    args = parser.parse_args()
    transcribe_audio(args.audio_path, args.model, args.language, args.translate) 