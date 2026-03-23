import numpy as np
import torch
from silero_vad import load_silero_vad, VADIterator

from config import Config


class VoiceActivityDetector:
    def __init__(self, config: Config):
        model = load_silero_vad()
        self._vad_iterator = VADIterator(
            model,
            threshold=config.vad_threshold,
            sampling_rate=config.sample_rate,
            min_silence_duration_ms=config.min_silence_duration_ms,
            speech_pad_ms=config.speech_pad_ms,
        )
        self._sample_rate = config.sample_rate
        self._min_speech_samples = int(
            config.min_speech_duration_ms * config.sample_rate / 1000
        )
        self._audio_buffer: list[np.ndarray] = []
        self._is_speech = False

    def process_chunk(
        self, audio_chunk_int16: np.ndarray
    ) -> tuple[str | None, np.ndarray | None]:
        audio_float32 = audio_chunk_int16.astype(np.float32) / 32768.0
        chunk_tensor = torch.from_numpy(audio_float32)

        speech_dict = self._vad_iterator(chunk_tensor, return_seconds=False)

        if speech_dict is not None and "start" in speech_dict:
            self._is_speech = True
            self._audio_buffer = [audio_chunk_int16.copy()]
            return ("speech_start", None)

        if self._is_speech:
            self._audio_buffer.append(audio_chunk_int16.copy())

        if speech_dict is not None and "end" in speech_dict:
            self._is_speech = False
            concatenated = np.concatenate(self._audio_buffer)
            self._audio_buffer = []

            if len(concatenated) < self._min_speech_samples:
                return (None, None)

            return ("speech_end", concatenated)

        return (None, None)

    def reset(self):
        self._vad_iterator.reset_states()
        self._audio_buffer = []
        self._is_speech = False
