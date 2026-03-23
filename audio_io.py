import queue

import numpy as np
import sounddevice as sd

from config import Config


class AudioIO:
    def __init__(self, config: Config):
        self.sample_rate = config.sample_rate
        self.channels = config.channels
        self.chunk_samples = config.chunk_samples
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[Audio] {status}")
        self._audio_queue.put(indata[:, 0].copy())

    def start_capture(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.chunk_samples,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop_capture(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_audio_chunk(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def play_audio(self, audio: np.ndarray, sample_rate: int):
        sd.play(audio, samplerate=sample_rate)
        sd.wait()

    def clear_queue(self):
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def close(self):
        self.stop_capture()
