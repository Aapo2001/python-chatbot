import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import voice_chatbot.platform_setup as platform_setup
from voice_chatbot.config import Config
from voice_chatbot.errors import AudioDependencyError


def _import_chatbot_module(
    monkeypatch,
    fresh_import,
    module_factory,
    stt_text="hei",
    llm_reply="moi",
    config=None,
    include_tts_module=True,
    llm_error=None,
):
    monkeypatch.setattr(platform_setup, "setup_cuda", lambda: None)

    state = {}

    class FakeAudioIO:
        def __init__(self, config):
            self.config = config
            self.started = False
            self.closed = False
            self.cleared = False
            self.play_calls = []
            self._chunks = [
                np.array([10, 11], dtype=np.int16),
                np.array([12, 13], dtype=np.int16),
                KeyboardInterrupt,
            ]
            state["audio"] = self

        def start_capture(self):
            self.started = True

        def get_audio_chunk(self, timeout=0.1):
            item = self._chunks.pop(0)
            if item is KeyboardInterrupt:
                raise KeyboardInterrupt
            return item

        def play_audio(self, audio, sample_rate):
            self.play_calls.append((audio.copy(), sample_rate))

        def clear_queue(self):
            self.cleared = True

        def close(self):
            self.closed = True

    class FakeVad:
        def __init__(self, config):
            self.events = [
                ("speech_start", None),
                ("speech_end", np.array([1, 2, 3], dtype=np.int16)),
            ]
            self.reset_calls = 0
            state["vad"] = self

        def process_chunk(self, chunk):
            return self.events.pop(0)

        def reset(self):
            self.reset_calls += 1

    class FakeStt:
        def __init__(self, config):
            self.calls = []
            state["stt"] = self

        def transcribe(self, audio):
            self.calls.append(audio.copy())
            return stt_text

    class FakeLlm:
        def __init__(self, config):
            self.calls = []
            state["llm"] = self

        def chat(self, text):
            self.calls.append(text)
            if llm_error is not None:
                raise llm_error
            return llm_reply

    class FakeTts:
        def __init__(self, config):
            self.calls = []
            state["tts"] = self

        def synthesize(self, text):
            self.calls.append(text)
            return np.array([0.25, -0.25], dtype=np.float32), 24000

    config_module = importlib.import_module("voice_chatbot.config")
    monkeypatch.setattr(
        config_module.Config,
        "load",
        classmethod(lambda cls: config if config is not None else Config()),
    )

    stub_modules = {
        "voice_chatbot.audio_io": module_factory(
            "voice_chatbot.audio_io", AudioIO=FakeAudioIO
        ),
        "voice_chatbot.vad": module_factory(
            "voice_chatbot.vad", VoiceActivityDetector=FakeVad
        ),
        "voice_chatbot.stt": module_factory("voice_chatbot.stt", SpeechToText=FakeStt),
        "voice_chatbot.llm": module_factory("voice_chatbot.llm", ChatLLM=FakeLlm),
    }
    if include_tts_module:
        stub_modules["voice_chatbot.tts_engine"] = module_factory(
            "voice_chatbot.tts_engine", TextToSpeech=FakeTts
        )

    module = fresh_import(
        "voice_chatbot.chatbot",
        stub_modules=stub_modules,
        clear_modules=["voice_chatbot.chatbot"],
    )
    return module, state


def test_run_processes_one_utterance_and_cleans_up(
    monkeypatch, fresh_import, module_factory
):
    module, state = _import_chatbot_module(monkeypatch, fresh_import, module_factory)
    chatbot = module.VoiceChatbot()

    chatbot.run()

    assert state["audio"].started is True
    assert state["stt"].calls and np.array_equal(
        state["stt"].calls[0], np.array([1, 2, 3], dtype=np.int16)
    )
    assert state["llm"].calls == ["hei"]
    assert state["tts"].calls == ["moi"]
    assert state["audio"].cleared is True
    assert state["vad"].reset_calls == 1
    assert state["audio"].closed is True
    assert state["audio"].play_calls[0][1] == 24000


def test_run_skips_llm_and_tts_when_transcript_is_blank(
    monkeypatch, fresh_import, module_factory
):
    module, state = _import_chatbot_module(
        monkeypatch,
        fresh_import,
        module_factory,
        stt_text="   ",
    )
    chatbot = module.VoiceChatbot()

    chatbot.run()

    assert state["llm"].calls == []
    assert state["tts"].calls == []
    assert state["audio"].play_calls == []
    assert state["audio"].cleared is False
    assert state["vad"].reset_calls == 0
    assert state["audio"].closed is True


def test_tts_is_not_required_when_disabled(monkeypatch, fresh_import, module_factory):
    module, state = _import_chatbot_module(
        monkeypatch,
        fresh_import,
        module_factory,
        config=Config(tts_enabled=False),
        include_tts_module=False,
    )

    chatbot = module.VoiceChatbot()
    chatbot.run()

    assert "tts" not in state
    assert state["audio"].play_calls == []
    assert state["audio"].closed is True


def test_audio_is_closed_when_pipeline_raises(
    monkeypatch, fresh_import, module_factory
):
    module, state = _import_chatbot_module(
        monkeypatch,
        fresh_import,
        module_factory,
        llm_error=RuntimeError("boom"),
    )
    chatbot = module.VoiceChatbot()

    with pytest.raises(RuntimeError, match="boom"):
        chatbot.run()

    assert state["audio"].closed is True


def test_main_prints_friendly_audio_dependency_error(
    monkeypatch, fresh_import, module_factory, capsys
):
    monkeypatch.setattr(platform_setup, "setup_cuda", lambda: None)

    config_module = importlib.import_module("voice_chatbot.config")
    monkeypatch.setattr(config_module.Config, "load", classmethod(lambda cls: Config()))

    class FailingAudioIO:
        def __init__(self, config):
            raise AudioDependencyError("PortAudio puuttuu")

    module = fresh_import(
        "voice_chatbot.chatbot",
        stub_modules={
            "voice_chatbot.audio_io": module_factory(
                "voice_chatbot.audio_io", AudioIO=FailingAudioIO
            )
        },
        clear_modules=["voice_chatbot.chatbot"],
    )

    module.main()

    output = capsys.readouterr().out
    assert "VIRHE:" in output
    assert "PortAudio puuttuu" in output
