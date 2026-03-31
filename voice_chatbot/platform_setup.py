"""
Platform-specific setup that must run before CUDA or Qt imports.

This module consolidates three environment fixups that were previously
duplicated across every entry point:

1. **CUDA DLL directories** – ``os.add_dll_directory`` for the CUDA
   toolkit so that ``torch``, ``llama_cpp``, and ``faster_whisper`` can
   find ``cublas64_*.dll`` etc.
2. **PySide6 / Robostack DLL conflict** – force Windows to prefer the
   pip-installed PySide6 Qt runtime over the one shipped by Robostack.
3. **espeak-ng PATH** – append the default Windows install directory
   so Coqui TTS can find the phonemiser backend.

Usage — call the functions you need at the **top** of each entry point,
**before** importing any CUDA-backed or Qt library::

    from voice_chatbot.platform_setup import setup_cuda, setup_pyside6

    setup_cuda()      # before torch / llama_cpp / faster_whisper
    setup_pyside6()   # before PySide6 widget imports
"""

import os
import sys
from pathlib import Path


def setup_cuda() -> None:
    """Register CUDA DLL directories on Windows.

    Reads ``CUDA_PATH`` (default ``D:\\cuda``) and adds ``bin/x64`` and
    ``bin`` subdirectories to the DLL search path.  Safe to call on
    Linux (no-ops if ``os.add_dll_directory`` is unavailable).
    """
    if not hasattr(os, "add_dll_directory"):
        return
    cuda_path = os.environ.get("CUDA_PATH", r"D:\cuda")
    for subdir in ("bin/x64", "bin"):
        p = os.path.join(cuda_path, subdir)
        if os.path.isdir(p):
            os.add_dll_directory(p)
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")


def setup_pyside6() -> None:
    """Fix PySide6 / Robostack Qt DLL conflicts on Windows.

    When both pip-installed PySide6 and conda-installed Qt (from
    Robostack) are present, Windows may load the wrong platform plugin
    and crash.  This function forces the PySide6 DLL directory onto
    ``PATH`` and overrides ``QT_PLUGIN_PATH``.
    """
    if not hasattr(os, "add_dll_directory"):
        return
    site_packages = [Path(p) for p in sys.path if "site-packages" in p]
    pyside_dir = None
    for pkg_name in ("PySide6", "shiboken6"):
        for base in site_packages:
            dll_dir = base / pkg_name
            if dll_dir.is_dir():
                os.add_dll_directory(str(dll_dir))
                os.environ["PATH"] = (
                    str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
                )
                if pkg_name == "PySide6":
                    pyside_dir = dll_dir

    if pyside_dir is not None:
        plugins_dir = pyside_dir / "plugins"
        platforms_dir = plugins_dir / "platforms"
        os.environ["QT_PLUGIN_PATH"] = str(plugins_dir)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_dir)


def setup_espeak() -> None:
    """Add the default eSpeak NG install directory to PATH on Windows.

    Required by Coqui TTS VITS models for phoneme conversion.
    """
    espeak_dir = os.path.join(
        os.environ.get("ProgramFiles", r"C:\Program Files"), "eSpeak NG"
    )
    path_value = os.environ.get("PATH", "")
    if os.path.isdir(espeak_dir) and espeak_dir not in path_value:
        os.environ["PATH"] = (
            espeak_dir if not path_value else espeak_dir + os.pathsep + path_value
        )


def is_wsl() -> bool:
    """Return True if running inside WSL (any version)."""
    try:
        with open("/proc/version") as f:
            content = f.read().lower()
        return "microsoft" in content or "wsl" in content
    except OSError:
        return False


def get_wsl_version() -> int:
    """Return WSL version: 2, 1, or 0 (not WSL)."""
    if not is_wsl():
        return 0
    # WSL2 has a real Linux kernel — /run/user/ is created by systemd/logind
    if os.path.isdir("/run/user") and os.environ.get("WSL_DISTRO_NAME"):
        return 2
    if os.environ.get("WSL_DISTRO_NAME"):
        return 1
    return 2  # /proc/version says microsoft but no distro var — assume WSL2


def setup_wsl_audio() -> None:
    """Configure ALSA → PulseAudio routing for WSL2/WSLg.

    Writes ~/.asoundrc if it does not already exist (or lacks pulse routing).
    Safe to call on non-WSL systems (no-ops immediately).
    Must be called before any sounddevice import (PortAudio probes ALSA
    at library load time).
    """
    import warnings

    if not is_wsl():
        return

    if get_wsl_version() == 1:
        warnings.warn(
            "[WSL Audio] WSL1 does not support audio hardware. "
            "Upgrade to WSL2 with WSLg for microphone/speaker access.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    if not os.environ.get("PULSE_SERVER"):
        warnings.warn(
            "[WSL Audio] PULSE_SERVER is not set — WSLg may not be active. "
            "Audio will likely fail. Ensure you are running WSL2 with WSLg "
            "(Windows 11 build 22000+ or Windows 10 with WSLg preview). "
            "If needed, run: export PULSE_SERVER=/run/user/$(id -u)/pulse/native",
            RuntimeWarning,
            stacklevel=2,
        )

    asoundrc = Path.home() / ".asoundrc"
    marker = "pcm.default pulse"
    content = (
        "# Added by voice-chatbot WSL audio setup\n"
        "# Routes ALSA default device to PulseAudio (WSLg).\n"
        "pcm.default pulse\n"
        "ctl.default pulse\n"
    )

    if asoundrc.exists():
        existing = asoundrc.read_text(encoding="utf-8", errors="replace")
        if marker in existing:
            return  # already configured, idempotent
        warnings.warn(
            f"[WSL Audio] ~/.asoundrc already exists with custom content.\n"
            f"Add the following lines manually to enable PulseAudio routing:\n\n{content}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    asoundrc.write_text(content, encoding="utf-8")
    print("[WSL Audio] Wrote ~/.asoundrc to route ALSA → PulseAudio (WSLg).")
