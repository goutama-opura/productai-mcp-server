import pyaudio
import wave
import threading
import time
import json
import numpy as np
import requests

from utils.engine import WhisperEngine
from utils.audio_utils import load_audio, normalize_audio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

frames = []
recording = True

def record_audio():
    global frames, recording
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("🎙️ Recording... Press ENTER to stop.")
    while recording:
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save file
    wf = wave.open("output.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("✅ Recording finished and saved as output.wav")


# Start recording in a separate thread
t = threading.Thread(target=record_audio)
t.start()

# Wait for user to press ENTER
input()
recording = False
t.join()

# --- Send to API ---
api_url = "http://127.0.0.1:8000/transcribe"
with open("output.wav", "rb") as f:
    files = {"file": ("output.wav", f, "audio/wav")}
    response = requests.post(api_url, files=files)

if response.status_code == 200:
    data = response.json()
    print("\n=== Transcribed Text from API ===")
    print(data["transcription"])
    print("\n=== Diagnostics from API ===")
    print(json.dumps(data["diagnostics"], indent=2))
else:
    print("API error:", response.status_code, response.text)