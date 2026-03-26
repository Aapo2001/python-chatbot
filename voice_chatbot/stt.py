"""
Speech-to-text transcription using `faster-whisper <https://github.com/SYSTRAN/faster-whisper>`_.

``faster-whisper`` is a CTranslate2 re-implementation of OpenAI's Whisper
that supports CUDA acceleration and is significantly faster than the
original PyTorch-based model at equivalent accuracy.

Usage::

    stt = SpeechToText(config)
    text = stt.transcribe(audio_int16)
"""

import inspect

import numpy as np
import torch
from faster_whisper import WhisperModel

from .config import Config


class SpeechToText:
    """Whisper-based speech-to-text engine.

    Loads the model once during construction.  All subsequent calls to
    :meth:`transcribe` reuse the same model instance.

    Args:
        config: Shared :class:`Config` – reads ``whisper_model``,
                ``whisper_gpu``, ``whisper_n_threads``, and ``language``.
    """

    def __init__(self, config: Config):
        self._language = config.language
        use_gpu = config.whisper_gpu and torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"
        compute_type = "float16" if use_gpu else "int8"
        print(
            f"[STT] Loading Whisper model '{config.whisper_model}' on {device} "
            f"(compute_type: {compute_type}, language: {self._language})..."
        )
        model_kwargs = {
            "device": device,
            "cpu_threads": config.whisper_n_threads,
        }
        if "compute_type" in inspect.signature(WhisperModel).parameters:
            model_kwargs["compute_type"] = compute_type
        self._model = WhisperModel(config.whisper_model, **model_kwargs)
        print("[STT] Whisper model loaded.")

    def transcribe(self, audio_int16: np.ndarray) -> str:
        """Transcribe an int16 audio array to text.

        Args:
            audio_int16: 1-D int16 numpy array captured by :class:`AudioIO`.

        Returns:
            The transcribed text with segments joined by spaces and
            leading/trailing whitespace stripped.  May be empty if
            Whisper produced no output.
        """
        # Whisper expects float32 audio normalised to [-1, 1].
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio_float32, language=self._language)
        text = " ".join(seg.text for seg in segments).strip()
        return text
