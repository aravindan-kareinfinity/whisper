# Whisper Local Transcription

This project demonstrates how to use OpenAI's Whisper model locally for audio transcription.

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- Git
- pip (Python package manager)

## Installation

1. Install FFmpeg:
   - Windows: Download from https://ffmpeg.org/download.html
   - Linux: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Basic usage:
```bash
python whisper_demo.py path/to/your/audio.mp3
```

Advanced options:
```bash
# Use a specific model (tiny, base, small, medium, large)
python whisper_demo.py audio.mp3 --model medium

# Specify language
python whisper_demo.py audio.mp3 --language en

# Translate to English
python whisper_demo.py audio.mp3 --translate
```

## Model Comparison

| Model  | Size  | Speed     | Accuracy |
|--------|-------|-----------|----------|
| tiny   | 39M   | Fastest   | Lowest   |
| base   | 74M   | Fast      | Better   |
| small  | 244M  | Medium    | Good     |
| medium | 769M  | Slower    | Very Good|
| large  | 1550M | Slowest   | Best     |

## Notes

- The first time you run the script, it will download the selected model
- GPU acceleration is automatically used if available
- Supported audio formats: MP3, WAV, M4A, etc. (any format supported by FFmpeg) 