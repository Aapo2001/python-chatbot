#!/usr/bin/env bash
set -euo pipefail

# ── WSL audio dependency installation ────────────────────────────────────────
if grep -qi "microsoft\|wsl" /proc/version 2>/dev/null; then
    echo "[WSL] WSL environment detected."
    if command -v apt-get >/dev/null 2>&1; then
        echo "[WSL] Installing PulseAudio ALSA plugin for WSLg audio support..."
        sudo apt-get update -qq
        sudo apt-get install -y --no-install-recommends \
            pulseaudio-utils \
            libasound2-plugins
        echo "[WSL] PulseAudio ALSA plugin installed."
    else
        echo "[WSL] WARNING: apt-get not found. Install 'libasound2-plugins' manually."
    fi
fi

echo "[pixi] Installing Python dependencies into the Pixi environment..."

python -m pip install --upgrade pip wheel

echo "[pixi] Installing PyTorch..."
python -m pip install --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio

export GGML_CUDA="${GGML_CUDA:-1}"
export CMAKE_ARGS="${CMAKE_ARGS:--DGGML_CUDA=on}"

echo "[pixi] Installing llama-cpp-python..."
python -m pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

echo "[pixi] Installing faster-whisper..."
python -m pip install --upgrade --force-reinstall faster-whisper --no-cache-dir

echo "[pixi] Installing remaining project requirements..."
python -m pip uninstall -y PySide6 PySide6_Addons PySide6_Essentials shiboken6 >/dev/null 2>&1 || true
python -m pip install --upgrade -r requirements.txt

echo "[pixi] Enforcing setuptools compatibility for ROS colcon builds..."
python tools/ensure_setuptools_compat.py

echo "[pixi] Linux Python dependency bootstrap complete."
