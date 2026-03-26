import numpy as np
import pytest

from voice_chatbot.config import Config

from .conftest import import_fresh, install_module, make_module


class FakeTTSModel:
    init_calls = []

    def __init__(self, **kwargs):
        type(self).init_calls.append(kwargs)
        self.to_calls = []
        self.synthesizer = make_module("synth", output_sample_rate=22050)
        type(self).instance = self

    def to(self, device):
        self.to_calls.append(device)
        return self

    def tts(self, text):
        self.last_text = text
        return [0.1, -0.2, 0.3]


def load_tts_module(monkeypatch, cuda_available=False):
    calls = {"setup_espeak": 0}
    FakeTTSModel.init_calls = []
    install_module(
        monkeypatch,
        "voice_chatbot.platform_setup",
        make_module(
            "voice_chatbot.platform_setup",
            setup_espeak=lambda: calls.__setitem__("setup_espeak", calls["setup_espeak"] + 1),
        ),
    )
    install_module(
        monkeypatch,
        "torch",
        make_module(
            "torch",
            device=lambda name: f"device:{name}",
            cuda=make_module("torch.cuda", is_available=lambda: cuda_available),
        ),
    )
    install_module(monkeypatch, "TTS", make_module("TTS"))
    install_module(monkeypatch, "TTS.api", make_module("TTS.api", TTS=FakeTTSModel))
    return import_fresh("voice_chatbot.tts_engine"), calls


def test_local_model_path_is_preferred_when_files_exist(monkeypatch, tmp_path):
    module, calls = load_tts_module(monkeypatch, cuda_available=True)
    model_path = tmp_path / "model.pth"
    config_path = tmp_path / "config.json"
    model_path.write_text("model", encoding="utf-8")
    config_path.write_text("{}", encoding="utf-8")

    tts = module.TextToSpeech(
        Config(
            tts_model="zoo-model",
            tts_model_path=str(model_path),
            tts_config_path=str(config_path),
            tts_gpu=True,
        )
    )

    assert calls["setup_espeak"] == 1
    assert FakeTTSModel.init_calls == [
        {"model_path": str(model_path), "config_path": str(config_path)}
    ]
    assert FakeTTSModel.instance.to_calls == ["device:cuda"]
    assert tts._sample_rate == 22050


def test_model_zoo_branch_is_used_when_local_files_are_missing(monkeypatch):
    module, _ = load_tts_module(monkeypatch, cuda_available=False)

    tts = module.TextToSpeech(Config(tts_model="tts_models/demo", tts_gpu=True))

    assert FakeTTSModel.init_calls == [{"model_name": "tts_models/demo"}]
    assert FakeTTSModel.instance.to_calls == ["device:cpu"]
    audio, sample_rate = tts.synthesize("Hei maailma")
    assert sample_rate == 22050
    assert audio.dtype == np.float32
    assert np.allclose(audio, np.array([0.1, -0.2, 0.3], dtype=np.float32))
    assert FakeTTSModel.instance.last_text == "Hei maailma"


def test_missing_synthesizer_raises_runtime_error(monkeypatch):
    load_tts_module(monkeypatch)

    class BrokenTTS(FakeTTSModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.synthesizer = None

    install_module(monkeypatch, "TTS.api", make_module("TTS.api", TTS=BrokenTTS))
    module = import_fresh("voice_chatbot.tts_engine")

    with pytest.raises(RuntimeError, match="synthesizer"):
        module.TextToSpeech(Config())
