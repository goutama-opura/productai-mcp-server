import whisper
import time
import os
import numpy as np
import soundfile as sf
import librosa




class WhisperEngine:
    def __init__(self, model_name="base"):
        self.model_name = model_name
        try:
            self.model = whisper.load_model(model_name)
            self.diagnostics = {"model_loaded": True, "model_name": model_name}
        except Exception as e:
            self.model = None
            self.diagnostics = {
                "model_loaded": False,
                "model_name": model_name,
                "error": str(e)
            }

    def transcribe(self, audio_input):
        """Transcribe audio with extended diagnostics."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        self.diagnostics["audio_file"] = audio_input

        # --- File validation ---
        if not os.path.exists(audio_input):
            self.diagnostics["file_error"] = "File not found"
            return ""

        # Duration
        try:
            audio_duration = self._get_audio_duration(audio_input)
            self.diagnostics["audio_duration_sec"] = audio_duration
        except Exception as e:
            self.diagnostics["audio_duration_error"] = str(e)

        # Noise check (SNR)
        try:
            y, sr = librosa.load(audio_input, sr=None)
            snr = self._calculate_snr(y)
            self.diagnostics["snr_db"] = round(snr, 2)
        except Exception as e:
            self.diagnostics["snr_error"] = str(e)
            y = None

        # --- Transcription ---
        start_time = time.time()
        try:
            result = self.model.transcribe(audio_input, verbose=False)
            text = result.get("text", "")

            self.diagnostics.update({
                "transcription_success": True,
                "transcription_time_sec": round(time.time() - start_time, 2),
                "text_length": len(text),
                "language": result.get("language"),
                "language_confidence": result.get("language_probability", None),
                "mean_logprob": np.mean([s.get("avg_logprob", 0) for s in result.get("segments", [])]) if result.get("segments") else None,
                "compression_ratio": np.mean([s.get("compression_ratio", 0) for s in result.get("segments", [])]) if result.get("segments") else None,
                "mean_no_speech_prob": np.mean([s.get("no_speech_prob", 0) for s in result.get("segments", [])]) if result.get("segments") else None,
                "speech_ratio": self._calculate_speech_ratio(y) if y is not None else None
            })

            # Accent flag
            if result.get("language") not in ["en", "hi"]:
                self.diagnostics["accent_warning"] = f"Detected language: {result.get('language')}"

            return text

        except Exception as e:
            self.diagnostics["transcription_success"] = False
            self.diagnostics["error"] = str(e)
            return ""

    def get_diagnostics(self):
        return self.diagnostics

    def _get_audio_duration(self, audio_path):
        with sf.SoundFile(audio_path) as f:
            return f.frames / f.samplerate

    def _calculate_snr(self, signal):
        rms_signal = np.sqrt(np.mean(signal**2))
        noise = signal - np.mean(signal)
        rms_noise = np.sqrt(np.mean(noise**2))
        return float("inf") if rms_noise == 0 else 20 * np.log10(rms_signal / rms_noise)

    def _calculate_speech_ratio(self, signal, threshold=0.01):
        """Rough estimate of speech vs silence."""
        energy = np.abs(signal)
        voiced = np.sum(energy > threshold)
        return voiced / len(signal)