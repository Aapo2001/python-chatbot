import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import voice_chatbot.platform_setup as platform_setup
from voice_chatbot.config import Config


def _import_setup_models_module(monkeypatch, fresh_import):
    monkeypatch.setattr(platform_setup, "setup_cuda", lambda: None)
    return fresh_import(
        "voice_chatbot.setup_models",
        clear_modules=["voice_chatbot.setup_models"],
    )


def test_check_cuda_reports_available_gpu(
    monkeypatch, fresh_import, module_factory, capsys
):
    module = _import_setup_models_module(monkeypatch, fresh_import)
    torch_module = module_factory(
        "torch",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index: "Fake GPU",
        ),
        version=SimpleNamespace(cuda="12.8"),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    module.check_cuda()

    out = capsys.readouterr().out
    assert "CUDA is available: Fake GPU" in out
    assert "CUDA version: 12.8" in out


def test_check_cuda_exits_when_torch_is_missing(monkeypatch, fresh_import, capsys):
    module = _import_setup_models_module(monkeypatch, fresh_import)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("torch missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit) as exc:
        module.check_cuda()

    assert exc.value.code == 1
    assert "PyTorch is not installed" in capsys.readouterr().out


def test_setup_llm_skips_download_when_model_exists(
    monkeypatch, fresh_import, capsys, tmp_path
):
    module = _import_setup_models_module(monkeypatch, fresh_import)
    model_path = tmp_path / "models" / "existing.gguf"
    model_path.parent.mkdir()
    model_path.write_bytes(b"x" * 2048)
    config = Config(
        models_dir=str(model_path.parent),
        llm_model_path=str(model_path),
        llm_filename="existing.gguf",
    )

    module.setup_llm(config)

    out = capsys.readouterr().out
    assert "LLM model already exists" in out
    assert model_path.exists()


def test_setup_llm_downloads_and_renames_file(
    monkeypatch, fresh_import, module_factory, tmp_path
):
    module = _import_setup_models_module(monkeypatch, fresh_import)
    download_path = tmp_path / "models" / "downloaded.gguf"
    target_path = tmp_path / "models" / "final.gguf"
    config = Config(
        models_dir=str(target_path.parent),
        llm_model_path=str(target_path),
        llm_repo_id="repo/test",
        llm_filename="final.gguf",
    )

    def fake_download(repo_id, filename, local_dir):
        assert repo_id == "repo/test"
        assert filename == "final.gguf"
        assert local_dir == str(target_path.parent)
        download_path.parent.mkdir(parents=True, exist_ok=True)
        download_path.write_text("model", encoding="utf-8")
        return str(download_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        module_factory("huggingface_hub", hf_hub_download=fake_download),
    )

    module.setup_llm(config)

    assert target_path.exists()
    assert target_path.read_text(encoding="utf-8") == "model"
    assert not download_path.exists()


def test_setup_llm_exits_with_manual_instructions_on_error(
    monkeypatch, fresh_import, module_factory, capsys, tmp_path
):
    module = _import_setup_models_module(monkeypatch, fresh_import)
    config = Config(
        models_dir=str(tmp_path / "models"),
        llm_model_path=str(tmp_path / "models" / "final.gguf"),
        llm_repo_id="repo/test",
        llm_filename="final.gguf",
    )

    def fake_download(repo_id, filename, local_dir):
        raise RuntimeError("download failed")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        module_factory("huggingface_hub", hf_hub_download=fake_download),
    )

    with pytest.raises(SystemExit) as exc:
        module.setup_llm(config)

    out = capsys.readouterr().out
    assert exc.value.code == 1
    assert "ERROR downloading LLM" in out
    assert "https://huggingface.co/repo/test" in out
    assert str(Path(config.llm_model_path)) in out


def test_main_runs_setup_steps_in_order(monkeypatch, fresh_import):
    module = _import_setup_models_module(monkeypatch, fresh_import)
    config = Config()
    calls = []

    monkeypatch.setattr(module.Config, "load", classmethod(lambda cls: config))
    monkeypatch.setattr(module, "check_cuda", lambda: calls.append("cuda"))
    monkeypatch.setattr(module, "setup_vad", lambda: calls.append("vad"))
    monkeypatch.setattr(
        module, "setup_whisper", lambda cfg: calls.append(("whisper", cfg))
    )
    monkeypatch.setattr(module, "setup_llm", lambda cfg: calls.append(("llm", cfg)))
    monkeypatch.setattr(module, "setup_tts", lambda cfg: calls.append(("tts", cfg)))

    module.main()

    assert calls == [
        "cuda",
        "vad",
        ("whisper", config),
        ("llm", config),
        ("tts", config),
    ]
