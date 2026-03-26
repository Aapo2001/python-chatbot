import numpy as np

from voice_chatbot.config import Config

from .conftest import import_fresh, install_module, make_module


class Segment:
    def __init__(self, text):
        self.text = text


class FakeWhisperModel:
    init_calls = []
    segments = []

    def __init__(self, model_name, device, cpu_threads):
        type(self).init_calls.append(
            {
                "model_name": model_name,
                "device": device,
                "cpu_threads": cpu_threads,
            }
        )
        self.transcribe_calls = []
        type(self).instance = self

    def transcribe(self, audio, language):
        self.transcribe_calls.append({"audio": audio, "language": language})
        return type(self).segments, {"ignored": True}


def load_stt_module(monkeypatch):
    FakeWhisperModel.init_calls = []
    FakeWhisperModel.segments = []
    install_module(
        monkeypatch,
        "faster_whisper",
        make_module("faster_whisper", WhisperModel=FakeWhisperModel),
    )
    return import_fresh("voice_chatbot.stt")


def test_initialization_uses_cuda_when_enabled(monkeypatch):
    module = load_stt_module(monkeypatch)

    module.SpeechToText(
        Config(language="en", whisper_model="small", whisper_gpu=True, whisper_n_threads=8)
    )

    assert FakeWhisperModel.init_calls == [
        {"model_name": "small", "device": "cuda", "cpu_threads": 8}
    ]


def test_transcribe_normalizes_audio_and_joins_segments(monkeypatch):
    module = load_stt_module(monkeypatch)
    FakeWhisperModel.segments = [Segment("  hello"), Segment("world  ")]
    stt = module.SpeechToText(Config(language="en", whisper_gpu=False))
    audio = np.array([0, 16384, -32768], dtype=np.int16)

    text = stt.transcribe(audio)

    call = FakeWhisperModel.instance.transcribe_calls[0]
    assert np.allclose(call["audio"], np.array([0.0, 0.5, -1.0], dtype=np.float32))
    assert call["audio"].dtype == np.float32
    assert call["language"] == "en"
    assert text == "hello world"
