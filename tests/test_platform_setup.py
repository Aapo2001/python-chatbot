import os
from pathlib import Path

import voice_chatbot.platform_setup as platform_setup


def test_setup_cuda_registers_existing_cuda_dirs(monkeypatch, tmp_path):
    cuda_root = tmp_path / "cuda"
    bin_x64 = cuda_root / "bin" / "x64"
    bin_root = cuda_root / "bin"
    bin_x64.mkdir(parents=True)
    bin_root.mkdir(exist_ok=True)

    added_dirs = []
    monkeypatch.setattr(os, "add_dll_directory", added_dirs.append, raising=False)
    monkeypatch.setenv("CUDA_PATH", str(cuda_root))
    monkeypatch.setenv("PATH", "")

    platform_setup.setup_cuda()

    assert [Path(path) for path in added_dirs] == [bin_x64, bin_root]
    path_entries = [entry for entry in os.environ["PATH"].split(os.pathsep) if entry]
    assert Path(str(bin_root)) in [Path(entry) for entry in path_entries]
    assert Path(str(bin_x64)) in [Path(entry) for entry in path_entries]


def test_setup_pyside6_prefers_pip_qt_plugins(monkeypatch, tmp_path):
    site_packages = tmp_path / "site-packages"
    pyside_dir = site_packages / "PySide6"
    shiboken_dir = site_packages / "shiboken6"
    (pyside_dir / "plugins" / "platforms").mkdir(parents=True)
    shiboken_dir.mkdir(parents=True)

    added_dirs = []
    monkeypatch.setattr(os, "add_dll_directory", added_dirs.append, raising=False)
    monkeypatch.setattr(platform_setup.sys, "path", [str(site_packages)])
    monkeypatch.setenv("PATH", "")

    platform_setup.setup_pyside6()

    assert str(pyside_dir) in added_dirs
    assert str(shiboken_dir) in added_dirs
    assert os.environ["QT_PLUGIN_PATH"] == str(pyside_dir / "plugins")
    assert os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] == str(
        pyside_dir / "plugins" / "platforms"
    )


def test_setup_espeak_populates_path_even_when_missing(monkeypatch, tmp_path):
    espeak_dir = tmp_path / "eSpeak NG"
    espeak_dir.mkdir()

    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.delenv("PATH", raising=False)

    platform_setup.setup_espeak()

    assert os.environ["PATH"] == str(espeak_dir)


def test_setup_espeak_does_not_duplicate_existing_path(monkeypatch, tmp_path):
    espeak_dir = tmp_path / "eSpeak NG"
    espeak_dir.mkdir()
    original_path = os.pathsep.join([str(espeak_dir), "C:\\Other"])

    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.setenv("PATH", original_path)

    platform_setup.setup_espeak()

    assert os.environ["PATH"] == original_path
