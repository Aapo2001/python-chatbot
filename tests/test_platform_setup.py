import os

from voice_chatbot import platform_setup


def test_setup_linux_cxx_runtime_preloads_conda_libraries(monkeypatch, tmp_path):
    prefix = tmp_path / "env"
    lib_dir = prefix / "lib"
    lib_dir.mkdir(parents=True)
    for name in ("libstdc++.so.6", "libgcc_s.so.1"):
        (lib_dir / name).write_text("", encoding="utf-8")

    loaded = []
    monkeypatch.setattr(platform_setup.sys, "platform", "linux")
    monkeypatch.setenv("CONDA_PREFIX", str(prefix))
    monkeypatch.setattr(
        platform_setup.ctypes,
        "CDLL",
        lambda path, mode=None: loaded.append((path, mode)),
    )
    monkeypatch.setattr(platform_setup, "_LINUX_CXX_RUNTIME_READY", False)

    platform_setup.setup_linux_cxx_runtime()

    assert loaded == [
        (str(lib_dir / "libstdc++.so.6"), platform_setup.ctypes.RTLD_GLOBAL),
        (str(lib_dir / "libgcc_s.so.1"), platform_setup.ctypes.RTLD_GLOBAL),
    ]
    assert platform_setup._LINUX_CXX_RUNTIME_READY is True


def test_setup_cuda_adds_existing_cuda_directories(monkeypatch):
    added = []
    cuda_root = r"C:\CUDA"
    cuda_bin_x64 = os.path.join(cuda_root, "bin/x64")
    cuda_bin = os.path.join(cuda_root, "bin")
    monkeypatch.setattr(
        platform_setup.os,
        "add_dll_directory",
        lambda path: added.append(path),
        raising=False,
    )
    monkeypatch.setenv("CUDA_PATH", cuda_root)
    monkeypatch.setenv("PATH", "tail")
    monkeypatch.setattr(
        platform_setup.os.path,
        "isdir",
        lambda path: path in {cuda_bin_x64, cuda_bin},
    )
    monkeypatch.setattr(platform_setup, "setup_linux_cxx_runtime", lambda: None)

    platform_setup.setup_cuda()

    assert added == [cuda_bin_x64, cuda_bin]
    assert os.environ["PATH"].startswith(cuda_bin + os.pathsep)


def test_setup_pyside6_updates_qt_plugin_paths(monkeypatch, tmp_path):
    added = []
    site_packages = tmp_path / "site-packages"
    pyside_dir = site_packages / "PySide6"
    (pyside_dir / "plugins" / "platforms").mkdir(parents=True)
    (site_packages / "shiboken6").mkdir(parents=True)

    monkeypatch.setattr(
        platform_setup.os,
        "add_dll_directory",
        lambda path: added.append(path),
        raising=False,
    )
    monkeypatch.setattr(platform_setup.sys, "path", [str(site_packages)])
    monkeypatch.setenv("PATH", "tail")

    platform_setup.setup_pyside6()

    assert str(pyside_dir) in added
    assert str(site_packages / "shiboken6") in added
    assert os.environ["QT_PLUGIN_PATH"] == str(pyside_dir / "plugins")
    assert os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] == str(
        pyside_dir / "plugins" / "platforms"
    )


def test_setup_espeak_prepends_default_install_dir(monkeypatch, tmp_path):
    program_files = tmp_path / "Program Files"
    espeak_dir = program_files / "eSpeak NG"
    espeak_dir.mkdir(parents=True)

    monkeypatch.setenv("ProgramFiles", str(program_files))
    monkeypatch.setenv("PATH", "tail")

    platform_setup.setup_espeak()

    assert os.environ["PATH"].startswith(str(espeak_dir) + os.pathsep)


def test_setup_espeak_does_not_duplicate_existing_path(monkeypatch, tmp_path):
    program_files = tmp_path / "Program Files"
    espeak_dir = program_files / "eSpeak NG"
    espeak_dir.mkdir(parents=True)
    existing_path = str(espeak_dir) + os.pathsep + "tail"

    monkeypatch.setenv("ProgramFiles", str(program_files))
    monkeypatch.setenv("PATH", existing_path)

    platform_setup.setup_espeak()

    assert os.environ["PATH"] == existing_path
