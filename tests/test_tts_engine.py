from types import SimpleNamespace

import numpy as np
import pytest

import voice_chatbot.platform_setup as platform_setup
from voice_chatbot.config import Config


def _build_tts_module(
    monkeypatch,
    fresh_import,
    module_factory,
    *,
    cuda_available=True,
    synthesizer=SimpleNamespace(output_sample_rate=22050),
):
    monkeypatch.setattr(platform_setup, "setup_espeak", lambda: None)
    state = {}

    class FakeRuntimeTTS:
        def __init__(self, **kwargs):
            state["init_kwargs"] = kwargs
            self.synthesizer = synthesizer

        def to(self, device):
            state["device"] = device
            return self

        def tts(self, text):
            state["text"] = text
            return [0.25, -0.5]

    torch_module = module_factory(
        "torch",
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        device=lambda name: f"device:{name}",
    )
    tts_api = module_factory("TTS.api", TTS=FakeRuntimeTTS)
    tts_package = module_factory("TTS", api=tts_api)

    module = fresh_import(
        "voice_chatbot.tts_engine",
        stub_modules={
            "torch": torch_module,
            "TTS": tts_package,
            "TTS.api": tts_api,
        },
        clear_modules=["voice_chatbot.tts_engine"],
    )
    return module, state


def test_tts_prefers_local_model_files(
    monkeypatch, fresh_import, module_factory, tmp_path
):
    module, state = _build_tts_module(monkeypatch, fresh_import, module_factory)
    model_path = tmp_path / "model.pth"
    config_path = tmp_path / "config.json"
    model_path.write_text("weights", encoding="utf-8")
    config_path.write_text("{}", encoding="utf-8")
    config = Config(
        tts_model_path=str(model_path),
        tts_config_path=str(config_path),
        tts_gpu=True,
    )

    tts = module.TextToSpeech(config)

    assert state["init_kwargs"] == {
        "model_path": str(model_path),
        "config_path": str(config_path),
    }
    assert state["device"] == "device:cuda"
    assert tts._sample_rate == 22050


def test_tts_falls_back_to_model_name_when_local_files_are_missing(
    monkeypatch, fresh_import, module_factory, tmp_path
):
    module, state = _build_tts_module(
        monkeypatch,
        fresh_import,
        module_factory,
        cuda_available=False,
    )
    config = Config(
        tts_model="tts_models/custom",
        tts_model_path=str(tmp_path / "missing-model.pth"),
        tts_config_path=str(tmp_path / "missing-config.json"),
        tts_gpu=True,
    )

    module.TextToSpeech(config)

    assert state["init_kwargs"] == {"model_name": "tts_models/custom"}
    assert state["device"] == "device:cpu"


def test_tts_raises_if_synthesizer_is_missing(
    monkeypatch, fresh_import, module_factory
):
    module, _ = _build_tts_module(
        monkeypatch,
        fresh_import,
        module_factory,
        synthesizer=None,
    )

    with pytest.raises(RuntimeError, match="synthesizer"):
        module.TextToSpeech(Config())


def test_synthesize_returns_float32_audio(monkeypatch, fresh_import, module_factory):
    module, state = _build_tts_module(monkeypatch, fresh_import, module_factory)
    tts = module.TextToSpeech(Config())

    audio, sample_rate = tts.synthesize("hei maailma")

    assert state["text"] == "hei maailma"
    assert audio.dtype == np.float32
    assert np.allclose(audio, np.array([0.25, -0.5], dtype=np.float32))
    assert sample_rate == 22050
