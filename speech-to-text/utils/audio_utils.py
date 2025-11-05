def load_audio(file_path):
    import librosa
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def normalize_audio(audio):
    import numpy as np
    return audio / np.max(np.abs(audio))