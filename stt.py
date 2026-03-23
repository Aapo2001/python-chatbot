import numpy as np
from pywhispercpp.model import Model

from config import Config


class SpeechToText:
    def __init__(self, config: Config):
        self._language = config.language
        print(f"[STT] Loading Whisper model '{config.whisper_model}' (language: {self._language})...")
        self._model = Model(
            config.whisper_model,
            n_threads=config.whisper_n_threads,
            language=self._language,
        )
        print("[STT] Whisper model loaded.")

    def transcribe(self, audio_int16: np.ndarray) -> str:
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        segments = self._model.transcribe(audio_float32, language=self._language)
        text = " ".join(seg.text for seg in segments).strip()
        return text
