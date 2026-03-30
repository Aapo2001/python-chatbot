import pytest

from voice_chatbot.config import Config

from .conftest import import_fresh, install_module, make_module


def load_setup_models_module(monkeypatch):
    calls = {"setup_cuda": 0}
    install_module(
        monkeypatch,
        "voice_chatbot.platform_setup",
        make_module(
            "voice_chatbot.platform_setup",
            setup_cuda=lambda: calls.__setitem__("setup_cuda", calls["setup_cuda"] + 1),
        ),
    )
    return import_fresh("voice_chatbot.setup_models"), calls


def test_check_cuda_reports_when_gpu_is_available(monkeypatch, capsys):
    module, calls = load_setup_models_module(monkeypatch)
    install_module(
        monkeypatch,
        "torch",
        make_module(
            "torch",
            cuda=make_module(
                "torch.cuda",
                is_available=lambda: True,
                get_device_name=lambda index: "Test GPU",
            ),
            version=make_module("torch.version", cuda="12.1"),
        ),
    )

    module.check_cuda()

    out = capsys.readouterr().out
    assert calls["setup_cuda"] == 1
    assert "CUDA is available: Test GPU" in out
    assert "CUDA version: 12.1" in out


def test_setup_llm_skips_download_when_model_exists(monkeypatch, tmp_path, capsys):
    module, _ = load_setup_models_module(monkeypatch)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"x" * 1024)
    config = Config(
        models_dir=str(tmp_path),
        llm_model_path=str(model_path),
    )

    module.setup_llm(config)

    out = capsys.readouterr().out
    assert "already exists" in out


def test_setup_llm_downloads_and_renames_model(monkeypatch, tmp_path):
    module, _ = load_setup_models_module(monkeypatch)
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    downloaded = download_dir / "downloaded.gguf"
    downloaded.write_text("weights", encoding="utf-8")
    target = tmp_path / "models" / "target.gguf"
    config = Config(
        models_dir=str(target.parent),
        llm_model_path=str(target),
    )
    calls = []

    install_module(
        monkeypatch,
        "huggingface_hub",
        make_module(
            "huggingface_hub",
            hf_hub_download=lambda **kwargs: calls.append(kwargs) or str(downloaded),
        ),
    )

    module.setup_llm(config)

    assert calls == [
        {
            "repo_id": module.DEFAULT_LLM_REPO_ID,
            "filename": module.DEFAULT_LLM_FILENAME,
            "local_dir": str(target.parent),
        }
    ]
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "weights"
    assert not downloaded.exists()


def test_setup_llm_exits_when_download_fails(monkeypatch, tmp_path):
    module, _ = load_setup_models_module(monkeypatch)
    config = Config(
        models_dir=str(tmp_path / "models"),
        llm_model_path=str(tmp_path / "models" / "missing.gguf"),
    )

    install_module(
        monkeypatch,
        "huggingface_hub",
        make_module(
            "huggingface_hub",
            hf_hub_download=lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        module.setup_llm(config)

    assert exc_info.value.code == 1


def test_main_runs_setup_steps_in_order(monkeypatch):
    module, _ = load_setup_models_module(monkeypatch)
    config = Config()
    calls = []

    monkeypatch.setattr(module.Config, "load", classmethod(lambda cls: config))
    monkeypatch.setattr(module, "check_cuda", lambda: calls.append("check_cuda"))
    monkeypatch.setattr(module, "setup_vad", lambda: calls.append("setup_vad"))
    monkeypatch.setattr(
        module, "setup_whisper", lambda cfg: calls.append(("setup_whisper", cfg))
    )
    monkeypatch.setattr(
        module, "setup_llm", lambda cfg: calls.append(("setup_llm", cfg))
    )
    monkeypatch.setattr(
        module, "setup_tts", lambda cfg: calls.append(("setup_tts", cfg))
    )

    module.main()

    assert calls == [
        "check_cuda",
        "setup_vad",
        ("setup_whisper", config),
        ("setup_llm", config),
        ("setup_tts", config),
    ]


def test_main_skips_tts_setup_when_disabled(monkeypatch):
    module, _ = load_setup_models_module(monkeypatch)
    config = Config(tts_enabled=False)
    calls = []

    monkeypatch.setattr(module.Config, "load", classmethod(lambda cls: config))
    monkeypatch.setattr(module, "check_cuda", lambda: calls.append("check_cuda"))
    monkeypatch.setattr(module, "setup_vad", lambda: calls.append("setup_vad"))
    monkeypatch.setattr(
        module, "setup_whisper", lambda cfg: calls.append(("setup_whisper", cfg))
    )
    monkeypatch.setattr(
        module, "setup_llm", lambda cfg: calls.append(("setup_llm", cfg))
    )
    monkeypatch.setattr(
        module, "setup_tts", lambda cfg: calls.append(("setup_tts", cfg))
    )

    module.main()

    assert calls == [
        "check_cuda",
        "setup_vad",
        ("setup_whisper", config),
        ("setup_llm", config),
    ]
