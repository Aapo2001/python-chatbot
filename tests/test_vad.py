from collections import deque

import numpy as np

from voice_chatbot.config import Config


def _build_vad_module(fresh_import, module_factory):
    state = {}

    class FakeVADIterator:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs
            self.responses = deque()
            self.calls = []
            self.reset_calls = 0
            state["iterator"] = self

        def __call__(self, tensor, return_seconds=False):
            self.calls.append((tensor.copy(), return_seconds))
            if self.responses:
                return self.responses.popleft()
            return None

        def reset_states(self):
            self.reset_calls += 1

    torch_module = module_factory("torch", from_numpy=lambda arr: arr)
    silero_vad = module_factory(
        "silero_vad",
        load_silero_vad=lambda: "vad-model",
        VADIterator=FakeVADIterator,
    )
    module = fresh_import(
        "voice_chatbot.vad",
        stub_modules={
            "torch": torch_module,
            "silero_vad": silero_vad,
        },
        clear_modules=["voice_chatbot.vad"],
    )
    return module, state


def test_vad_uses_energy_gate_before_speech_start(fresh_import, module_factory):
    module, state = _build_vad_module(fresh_import, module_factory)
    detector = module.VoiceActivityDetector(Config())
    quiet_chunk = np.zeros(512, dtype=np.int16)

    event, audio = detector.process_chunk(quiet_chunk)

    assert event is None
    assert audio is None
    assert state["iterator"].calls == []
    assert len(detector._pre_buffer) == 1


def test_vad_returns_concatenated_audio_with_prebuffer(fresh_import, module_factory):
    module, state = _build_vad_module(fresh_import, module_factory)
    config = Config(
        sample_rate=1000,
        chunk_samples=5,
        min_speech_duration_ms=10,
        vad_pre_buffer_ms=5,
    )
    detector = module.VoiceActivityDetector(config)
    state["iterator"].responses.extend([None, {"start": 5}, {"end": 15}])

    prebuffer = np.array([100, 100, 100, 100, 100], dtype=np.int16)
    start_chunk = np.array([200, 200, 200, 200, 200], dtype=np.int16)
    end_chunk = np.array([300, 300, 300, 300, 300], dtype=np.int16)

    assert detector.process_chunk(prebuffer) == (None, None)
    assert detector.process_chunk(start_chunk) == ("speech_start", None)
    event, audio = detector.process_chunk(end_chunk)

    assert event == "speech_end"
    assert np.array_equal(audio, np.concatenate([prebuffer, start_chunk, end_chunk]))


def test_vad_drops_short_utterances(fresh_import, module_factory):
    module, state = _build_vad_module(fresh_import, module_factory)
    config = Config(
        sample_rate=1000,
        chunk_samples=5,
        min_speech_duration_ms=30,
        vad_pre_buffer_ms=5,
    )
    detector = module.VoiceActivityDetector(config)
    state["iterator"].responses.extend([{"start": 5}, {"end": 10}])

    assert detector.process_chunk(np.full(5, 150, dtype=np.int16)) == (
        "speech_start",
        None,
    )
    event, audio = detector.process_chunk(np.full(5, 150, dtype=np.int16))

    assert event is None
    assert audio is None
    assert detector._is_speech is False
    assert detector._audio_buffer == []


def test_vad_reset_clears_state_and_resets_iterator(fresh_import, module_factory):
    module, state = _build_vad_module(fresh_import, module_factory)
    detector = module.VoiceActivityDetector(Config())
    detector._is_speech = True
    detector._audio_buffer = [np.array([1, 2], dtype=np.int16)]
    detector._pre_buffer.append(np.array([3, 4], dtype=np.int16))

    detector.reset()

    assert detector._is_speech is False
    assert detector._audio_buffer == []
    assert len(detector._pre_buffer) == 0
    assert state["iterator"].reset_calls == 1
