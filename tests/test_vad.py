import numpy as np

from voice_chatbot.config import Config

from .conftest import import_fresh, install_module, make_module


class FakeVADIterator:
    init_calls = []
    queued_responses = []

    def __init__(self, model, **kwargs):
        type(self).init_calls.append({"model": model, **kwargs})
        self.calls = []
        self.reset_called = False
        type(self).instance = self

    def __call__(self, tensor, return_seconds=False):
        self.calls.append({"tensor": tensor, "return_seconds": return_seconds})
        if type(self).queued_responses:
            return type(self).queued_responses.pop(0)
        return None

    def reset_states(self):
        self.reset_called = True


def load_vad_module(monkeypatch):
    FakeVADIterator.init_calls = []
    FakeVADIterator.queued_responses = []
    install_module(monkeypatch, "torch", make_module("torch", from_numpy=lambda array: array))
    install_module(
        monkeypatch,
        "silero_vad",
        make_module(
            "silero_vad",
            load_silero_vad=lambda: "model",
            VADIterator=FakeVADIterator,
        ),
    )
    return import_fresh("voice_chatbot.vad")


def test_silence_below_energy_floor_is_buffered_without_hitting_vad(monkeypatch):
    module = load_vad_module(monkeypatch)
    vad = module.VoiceActivityDetector(Config(sample_rate=1000, chunk_samples=4))
    silent_chunk = np.zeros(4, dtype=np.int16)

    event, audio = vad.process_chunk(silent_chunk)

    assert (event, audio) == (None, None)
    assert len(FakeVADIterator.instance.calls) == 0
    assert len(vad._pre_buffer) == 1


def test_speech_start_prepends_prebuffer(monkeypatch):
    module = load_vad_module(monkeypatch)
    vad = module.VoiceActivityDetector(
        Config(sample_rate=1000, chunk_samples=4, vad_pre_buffer_ms=8)
    )
    prebuffer_chunk = np.full(4, 200, dtype=np.int16)
    start_chunk = np.full(4, 300, dtype=np.int16)

    assert vad.process_chunk(prebuffer_chunk) == (None, None)
    FakeVADIterator.queued_responses = [{"start": 0}]

    event, audio = vad.process_chunk(start_chunk)

    assert (event, audio) == ("speech_start", None)
    assert len(vad._audio_buffer) == 2
    assert np.array_equal(vad._audio_buffer[0], prebuffer_chunk)
    assert np.array_equal(vad._audio_buffer[1], start_chunk)


def test_speech_end_returns_concatenated_audio(monkeypatch):
    module = load_vad_module(monkeypatch)
    vad = module.VoiceActivityDetector(
        Config(
            sample_rate=1000,
            chunk_samples=4,
            min_speech_duration_ms=1,
            vad_pre_buffer_ms=8,
        )
    )
    leading = np.full(4, 200, dtype=np.int16)
    start_chunk = np.full(4, 300, dtype=np.int16)
    end_chunk = np.full(4, 400, dtype=np.int16)

    assert vad.process_chunk(leading) == (None, None)
    FakeVADIterator.queued_responses = [{"start": 0}]
    assert vad.process_chunk(start_chunk) == ("speech_start", None)
    FakeVADIterator.queued_responses = [{"end": 4}]

    event, audio = vad.process_chunk(end_chunk)

    assert event == "speech_end"
    assert np.array_equal(audio, np.concatenate([leading, start_chunk, end_chunk]))
    assert vad._audio_buffer == []
    assert len(vad._pre_buffer) == 0


def test_short_utterance_is_discarded(monkeypatch):
    module = load_vad_module(monkeypatch)
    vad = module.VoiceActivityDetector(
        Config(sample_rate=1000, chunk_samples=4, min_speech_duration_ms=50)
    )
    start_chunk = np.full(4, 300, dtype=np.int16)
    end_chunk = np.full(4, 300, dtype=np.int16)

    FakeVADIterator.queued_responses = [{"start": 0}]
    assert vad.process_chunk(start_chunk) == ("speech_start", None)
    FakeVADIterator.queued_responses = [{"end": 4}]

    assert vad.process_chunk(end_chunk) == (None, None)


def test_reset_clears_state_and_resets_iterator(monkeypatch):
    module = load_vad_module(monkeypatch)
    vad = module.VoiceActivityDetector(Config())
    vad._audio_buffer = [np.array([1], dtype=np.int16)]
    vad._pre_buffer.append(np.array([2], dtype=np.int16))
    vad._is_speech = True

    vad.reset()

    assert vad._audio_buffer == []
    assert list(vad._pre_buffer) == []
    assert vad._is_speech is False
    assert FakeVADIterator.instance.reset_called is True
