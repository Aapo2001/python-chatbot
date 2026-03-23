from collections import deque

import numpy as np
import torch
from silero_vad import load_silero_vad, VADIterator

from config import Config

# RMS threshold below which audio is considered silence (int16 scale).
# Prevents VAD from triggering on low-level electrical noise.
# Set very low — Silero VAD itself handles speech/noise discrimination well,
# this only needs to block near-zero electrical noise.
_ENERGY_FLOOR_RMS = 30


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

        # Pre-buffer: keeps recent chunks so we capture audio *before*
        # Silero reports "speech_start".  Without this the first syllable
        # is always clipped, producing garbled Whisper output.
        pre_buf_chunks = max(
            1,
            int(config.vad_pre_buffer_ms * config.sample_rate
                / 1000 / config.chunk_samples),
        )
        self._pre_buffer: deque[np.ndarray] = deque(maxlen=pre_buf_chunks)

        self._audio_buffer: list[np.ndarray] = []
        self._is_speech = False

    def process_chunk(
        self, audio_chunk_int16: np.ndarray
    ) -> tuple[str | None, np.ndarray | None]:
        # Energy gate — skip near-silent frames so electrical noise
        # doesn't feed into Silero and cause phantom triggers.
        rms = np.sqrt(np.mean(audio_chunk_int16.astype(np.float64) ** 2))
        if rms < _ENERGY_FLOOR_RMS:
            self._pre_buffer.append(audio_chunk_int16.copy())
            if self._is_speech:
                # Still accumulate during speech (could be a brief quiet gap)
                self._audio_buffer.append(audio_chunk_int16.copy())
            return (None, None)

        audio_float32 = audio_chunk_int16.astype(np.float32) / 32768.0
        chunk_tensor = torch.from_numpy(audio_float32)

        speech_dict = self._vad_iterator(chunk_tensor, return_seconds=False)

        if speech_dict is not None and "start" in speech_dict:
            self._is_speech = True
            # Prepend the pre-buffer so the onset of speech is preserved
            self._audio_buffer = list(self._pre_buffer)
            self._audio_buffer.append(audio_chunk_int16.copy())
            self._pre_buffer.clear()
            return ("speech_start", None)

        if self._is_speech:
            self._audio_buffer.append(audio_chunk_int16.copy())

        if speech_dict is not None and "end" in speech_dict:
            self._is_speech = False
            concatenated = np.concatenate(self._audio_buffer)
            self._audio_buffer = []
            self._pre_buffer.clear()

            if len(concatenated) < self._min_speech_samples:
                return (None, None)

            return ("speech_end", concatenated)

        # Always feed the pre-buffer when not in speech
        if not self._is_speech:
            self._pre_buffer.append(audio_chunk_int16.copy())

        return (None, None)

    def reset(self):
        self._vad_iterator.reset_states()
        self._audio_buffer = []
        self._pre_buffer.clear()
        self._is_speech = False
