import numpy as np
import pytest

from voice_chatbot.config import Config
from voice_chatbot.errors import AudioDependencyError

from .conftest import import_fresh, install_module, make_module


class FakeInputStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def load_audio_module(monkeypatch):
    state = {"played": [], "waited": 0, "streams": []}

    def input_stream_factory(**kwargs):
        stream = FakeInputStream(**kwargs)
        state["streams"].append(stream)
        return stream

    install_module(
        monkeypatch,
        "sounddevice",
        make_module(
            "sounddevice",
            InputStream=input_stream_factory,
            play=lambda audio, samplerate: state["played"].append((audio, samplerate)),
            wait=lambda: state.__setitem__("waited", state["waited"] + 1),
        ),
    )
    return import_fresh("voice_chatbot.audio_io"), state


def test_start_capture_configures_and_starts_input_stream(monkeypatch):
    module, state = load_audio_module(monkeypatch)
    audio = module.AudioIO(Config(sample_rate=8000, channels=2, chunk_samples=256))

    audio.start_capture()

    assert len(state["streams"]) == 1
    stream = state["streams"][0]
    assert stream.started is True
    assert stream.kwargs["samplerate"] == 8000
    assert stream.kwargs["channels"] == 2
    assert stream.kwargs["dtype"] == "int16"
    assert stream.kwargs["blocksize"] == 256


def test_stop_capture_closes_existing_stream(monkeypatch):
    module, _ = load_audio_module(monkeypatch)
    audio = module.AudioIO(Config())
    stream = FakeInputStream()
    audio._stream = stream

    audio.stop_capture()

    assert stream.stopped is True
    assert stream.closed is True
    assert audio._stream is None


def test_audio_callback_copies_first_channel_into_queue(monkeypatch):
    module, _ = load_audio_module(monkeypatch)
    audio = module.AudioIO(Config())
    chunk = np.array([[1, 100], [2, 200], [3, 300]], dtype=np.int16)

    audio._audio_callback(chunk, None, None, None)
    chunk[0, 0] = 999

    queued = audio.get_audio_chunk(timeout=0.01)
    assert np.array_equal(queued, np.array([1, 2, 3], dtype=np.int16))


def test_get_audio_chunk_returns_none_when_queue_is_empty(monkeypatch):
    module, _ = load_audio_module(monkeypatch)
    audio = module.AudioIO(Config())

    assert audio.get_audio_chunk(timeout=0.01) is None


def test_clear_queue_drains_all_buffered_chunks(monkeypatch):
    module, _ = load_audio_module(monkeypatch)
    audio = module.AudioIO(Config())
    audio._audio_queue.put(np.array([1], dtype=np.int16))
    audio._audio_queue.put(np.array([2], dtype=np.int16))

    audio.clear_queue()

    assert audio.get_audio_chunk(timeout=0.01) is None


def test_play_audio_delegates_to_sounddevice(monkeypatch):
    module, state = load_audio_module(monkeypatch)
    audio = module.AudioIO(Config())
    samples = np.array([0.1, -0.1], dtype=np.float32)

    audio.play_audio(samples, 22050)

    assert state["played"] == [(samples, 22050)]
    assert state["waited"] == 1


def test_import_raises_friendly_error_when_portaudio_is_missing(monkeypatch):
    import importlib

    original_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(AudioDependencyError, match="PortAudio"):
        import_fresh("voice_chatbot.audio_io")
