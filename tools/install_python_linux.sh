#!/usr/bin/env bash
set -euo pipefail

echo "[pixi] Installing Python dependencies into the Pixi environment..."

python -m pip install --upgrade pip wheel

echo "[pixi] Installing PyTorch..."
python -m pip install --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio

export GGML_CUDA="${GGML_CUDA:-1}"
export CMAKE_ARGS="${CMAKE_ARGS:--DGGML_CUDA=on}"

echo "[pixi] Installing llama-cpp-python..."
python -m pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

echo "[pixi] Installing pywhispercpp..."
python -m pip install --upgrade --force-reinstall pywhispercpp --no-cache-dir

echo "[pixi] Installing remaining project requirements..."
python -m pip uninstall -y PySide6 PySide6_Addons PySide6_Essentials shiboken6 >/dev/null 2>&1 || true
python -m pip install --upgrade -r requirements.txt

echo "[pixi] Enforcing setuptools compatibility for ROS colcon builds..."
python tools/ensure_setuptools_compat.py

echo "[pixi] Linux Python dependency bootstrap complete."
