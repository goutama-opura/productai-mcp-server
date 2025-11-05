import warnings

# Suppress pkg_resources deprecation warning globally
warnings.filterwarnings(
    "ignore",
    message=".*pkg_resources is deprecated.*",
    category=DeprecationWarning,
)

import os
import json
import tempfile
import shutil
import sounddevice as sd
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# Your other imports
from utils.engine import WhisperEngine
from utils.audio_utils import load_audio, normalize_audio
from faster_whisper import WhisperModel

# Initialize engine
engine = WhisperEngine(model_name="base")

# Initialize faster whisper model
faster_model = WhisperModel("small", device="cpu", compute_type="int8")

app = FastAPI()

class TranscriptionResponse(BaseModel):
    success: bool
    transcription: str
    diagnostics: dict


def clean_diagnostics(diagnostics):
    import numpy as np
    def convert(obj):
        if isinstance(obj, (np.generic,)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    return convert(diagnostics)


@app.get("/")
def home():
    return {"message": "Voice → Text → API is running!"}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def stt_transcribe(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            audio, sr = load_audio(temp_path)
            audio = normalize_audio(audio)
        except Exception as e:
            print(f"[Warning] Audio preprocessing skipped: {e}")

        text = engine.transcribe(temp_path)
        diagnostics = clean_diagnostics(engine.get_diagnostics())
        os.remove(temp_path)

        return {"success": True, "transcription": text, "diagnostics": diagnostics}
    except Exception as e:
        return {"success": False, "error": str(e)}

