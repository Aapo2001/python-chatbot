"""
Text-to-speech synthesis using `Coqui TTS <https://github.com/coqui-ai/TTS>`_.

Supports two loading modes:

1. **Local model** – if ``tts_model_path`` (a ``.pth`` checkpoint) and
   ``tts_config_path`` both exist on disk, they are loaded directly.
   This is the preferred path for custom-trained VITS models.

2. **HuggingFace / Coqui model zoo** – otherwise the model identified
   by the ``tts_model`` string (e.g. ``tts_models/fi/css10/vits``) is
   downloaded and cached automatically by the Coqui TTS library.

.. note::

   On Windows, the ``espeak-ng`` phonemiser must be on ``PATH`` for
   most VITS models.  This module prepends the default eSpeak NG
   install directory to ``PATH`` before importing the TTS library.

Usage::

    tts = TextToSpeech(config)
    audio_array, sample_rate = tts.synthesize("Hei maailma!")
"""

import os

# ── espeak-ng path setup (must run before TTS import) ─────────────
# VITS models use espeak-ng for phoneme conversion.  On Windows the
# installer places it under "C:\\Program Files\\eSpeak NG" which is
# typically not on PATH inside a pixi/conda environment.
_espeak_dir = os.path.join(
    os.environ.get("ProgramFiles", r"C:\Program Files"), "eSpeak NG"
)
if os.path.isdir(_espeak_dir) and _espeak_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _espeak_dir + os.pathsep + os.environ["PATH"]

import numpy as np
import torch
from TTS.api import TTS

from config import Config


class TextToSpeech:
    """Coqui TTS wrapper with local-model and model-zoo support.

    Args:
        config: Shared :class:`Config` – reads ``tts_model_path``,
                ``tts_config_path``, ``tts_model``, and ``tts_gpu``.
    """

    def __init__(self, config: Config):
        model_path = config.tts_model_path
        config_path = config.tts_config_path
        device = torch.device(
            "cuda" if torch.cuda.is_available() & config.tts_gpu else "cpu"
        )

        # Prefer a local checkpoint + config if both files exist.
        if os.path.isfile(model_path) and os.path.isfile(config_path):
            print(
                f"[TTS] Loading local model '{model_path}' with config '{config_path}'..."
            )
            self._tts = TTS(
                model_path=model_path,
                config_path=config_path,
            ).to(device)
        else:
            # Fall back to the Coqui model zoo identifier.
            print(f"[TTS] Loading model '{config.tts_model}'...")
            self._tts = TTS(model_name=config.tts_model).to(device)

        if self._tts.synthesizer is None:
            raise RuntimeError("TTS synthesizer is not initialized.")
        self._sample_rate = self._tts.synthesizer.output_sample_rate
        print(f"[TTS] Model loaded (sample rate: {self._sample_rate} Hz).")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Convert text to a waveform.

        Args:
            text: The string to synthesise.

        Returns:
            A ``(audio, sample_rate)`` tuple where *audio* is a 1-D
            float32 numpy array and *sample_rate* is in Hz.
        """
        wav = self._tts.tts(text=text)
        audio = np.array(wav, dtype=np.float32)
        return audio, self._sample_rate
