import os
import argparse
import json
import requests
# from src.engine import WhisperEngine
from utils.engine import WhisperEngine
from utils.audio_utils import load_audio, normalize_audio


os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin"

def convert_speech_to_text_local(audio_file_path, model_name="base"):
    engine = WhisperEngine(model_name=model_name)
    transcribed_text = engine.transcribe(audio_file_path)
    diagnostics = engine.get_diagnostics()
    return transcribed_text, diagnostics

def convert_speech_to_text_api(audio_file_path, api_url="http://127.0.0.1:8000/transcribe"):
    with open(audio_file_path, "rb") as f:
        files = {"file": (audio_file_path, f, "audio/wav")}
        response = requests.post(api_url, files=files)
        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")
        data = response.json()
        return data["transcription"], data["diagnostics"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert speech to text using Whisper Engine or API.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file.")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size (tiny, base, small, medium, large).")
    parser.add_argument("--api", type=str, help="If provided, send request to FastAPI instead of running locally.")

    args = parser.parse_args()

    if args.api:
        print("Using FastAPI backend...")
        text, diagnostics = convert_speech_to_text_api(args.audio_file, api_url=args.api)
    else:
        print("Using local WhisperEngine...")
        text, diagnostics = convert_speech_to_text_local(args.audio_file, model_name=args.model)

    print("\n=== Transcribed Text ===")
    print(text)
    print("\n=== Diagnostics ===")
    print(json.dumps(diagnostics, indent=2))