from types import SimpleNamespace

import numpy as np

from voice_chatbot.config import Config


def _build_stt_module(fresh_import, module_factory):
    state = {}

    class Segment:
        def __init__(self, text):
            self.text = text

    class FakeWhisperModel:
        def __init__(self, model_name, device, cpu_threads):
            state["init"] = {
                "model_name": model_name,
                "device": device,
                "cpu_threads": cpu_threads,
            }

        def transcribe(self, audio, language):
            state["audio"] = audio.copy()
            state["language"] = language
            return [Segment(" Hei"), Segment(" maailma ")], {"language": language}

    faster_whisper = module_factory("faster_whisper", WhisperModel=FakeWhisperModel)
    module = fresh_import(
        "voice_chatbot.stt",
        stub_modules={"faster_whisper": faster_whisper},
        clear_modules=["voice_chatbot.stt"],
    )
    return module, state


def test_stt_initializes_model_with_requested_device(fresh_import, module_factory):
    module, state = _build_stt_module(fresh_import, module_factory)
    config = Config(whisper_model="small", whisper_gpu=False, whisper_n_threads=6)

    module.SpeechToText(config)

    assert state["init"] == {
        "model_name": "small",
        "device": "cpu",
        "cpu_threads": 6,
    }


def test_transcribe_normalizes_audio_and_joins_segments(fresh_import, module_factory):
    module, state = _build_stt_module(fresh_import, module_factory)
    stt = module.SpeechToText(Config(language="en"))
    audio = np.array([0, 16384, -32768], dtype=np.int16)

    text = stt.transcribe(audio)

    assert state["language"] == "en"
    assert state["audio"].dtype == np.float32
    assert np.allclose(state["audio"], np.array([0.0, 0.5, -1.0], dtype=np.float32))
    assert text == "Hei  maailma"
