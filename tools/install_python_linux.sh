#!/usr/bin/env bash
set -euo pipefail

echo "[pixi] Installing Python dependencies into the Pixi environment..."

python -m pip install --upgrade pip setuptools wheel

echo "[pixi] Installing PyTorch..."
python -m pip install --upgrade torch torchvision torchaudio

export GGML_CUDA="${GGML_CUDA:-1}"
export CMAKE_ARGS="${CMAKE_ARGS:--DGGML_CUDA=on}"

echo "[pixi] Installing llama-cpp-python..."
python -m pip install --upgrade llama-cpp-python --no-cache-dir

echo "[pixi] Installing pywhispercpp..."
python -m pip install --upgrade pywhispercpp --no-cache-dir

echo "[pixi] Installing remaining project requirements..."
python -m pip install --upgrade -r requirements.txt

echo "[pixi] Linux Python dependency bootstrap complete."
