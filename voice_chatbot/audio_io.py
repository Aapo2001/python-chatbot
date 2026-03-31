"""
Audio input/output via the ``sounddevice`` library.

Provides :class:`AudioIO`, a thin wrapper around a ``sounddevice``
input stream and its blocking playback function.  Audio chunks captured
from the microphone are placed into a thread-safe :class:`queue.Queue`
so that the voice-activity detector can consume them at its own pace.

Typical lifecycle::

    audio = AudioIO(config)
    audio.start_capture()
    while running:
        chunk = audio.get_audio_chunk()   # int16 numpy array or None
        ...
    audio.close()
"""

import importlib
import queue
import sys

import numpy as np

from .config import Config
from .errors import AudioDependencyError


def _load_sounddevice():
    """Import ``sounddevice`` and raise a user-facing error on missing backend libs."""
    try:
        return importlib.import_module("sounddevice")
    except ModuleNotFoundError as exc:
        raise AudioDependencyError(
            "Python-paketti 'sounddevice' puuttuu.\n"
            "Aja ensin: pixi run install-python-deps"
        ) from exc
    except OSError as exc:
        if "PortAudio library not found" not in str(exc):
            raise

        help_lines = ["PortAudio-kirjastoa ei loytynyt, joten audio I/O ei kaynnisty."]
        if sys.platform.startswith("linux"):
            help_lines.append("Pixi-asennuksessa korjaus on: pixi install")
            help_lines.append(
                "Jos kaytat pip-asennusta ilman pixia, asenna myos jarjestelman "
                "PortAudio-paketti (esim. libportaudio2)."
            )

        raise AudioDependencyError("\n".join(help_lines)) from exc


sd = _load_sounddevice()


class AudioIO:
    """Microphone capture and audio playback.

    The capture path uses a callback-based ``InputStream`` that writes
    fixed-size chunks (default 512 samples ≈ 32 ms at 16 kHz) into an
    internal queue.  Playback is synchronous (blocks until done) so that
    callers can clear the microphone buffer and reset VAD immediately
    after the assistant finishes speaking.

    Args:
        config: Shared :class:`Config` instance – reads ``sample_rate``,
                ``channels``, and ``chunk_samples``.
    """

    def __init__(self, config: Config):
        self.sample_rate = config.sample_rate
        self.channels = config.channels
        self.chunk_samples = config.chunk_samples
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None

    # ── Callback ──────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        """``sounddevice`` stream callback – pushes mono int16 chunks."""
        if status:
            print(f"[Audio] {status}")
        # Extract channel 0 and copy to decouple from the driver buffer.
        self._audio_queue.put(indata[:, 0].copy())

    # ── Capture control ───────────────────────────────────────────

    def start_capture(self):
        """Open the default input device and begin streaming."""
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.chunk_samples,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop_capture(self):
        """Stop and close the input stream (safe to call when already stopped)."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ── Queue access ──────────────────────────────────────────────

    def get_audio_chunk(self, timeout: float = 0.1) -> np.ndarray | None:
        """Return the next int16 audio chunk, or ``None`` on timeout."""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self):
        """Drain all buffered chunks.

        Called after TTS playback to discard audio that was recorded
        while the assistant was speaking — prevents self-triggering.
        """
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    # ── Playback ──────────────────────────────────────────────────

    def play_audio(self, audio: np.ndarray, sample_rate: int):
        """Play an audio array through the default output device (blocking)."""
        sd.play(audio, samplerate=sample_rate)
        sd.wait()

    # ── Lifecycle ─────────────────────────────────────────────────

    def close(self):
        """Release all audio resources."""
        self.stop_capture()
