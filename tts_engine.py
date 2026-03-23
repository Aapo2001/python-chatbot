import numpy as np
from TTS.api import TTS

from config import Config


class TextToSpeech:
    def __init__(self, config: Config):
        print(f"[TTS] Loading model '{config.tts_model}'...")
        self._tts = TTS(model_name=config.tts_model, gpu=config.tts_gpu)
        self._sample_rate = self._tts.synthesizer.output_sample_rate
        print(f"[TTS] Model loaded (sample rate: {self._sample_rate} Hz).")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        wav = self._tts.tts(text=text)
        audio = np.array(wav, dtype=np.float32)
        return audio, self._sample_rate
