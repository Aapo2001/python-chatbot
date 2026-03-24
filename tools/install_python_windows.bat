@echo off
setlocal

echo [pixi] Installing Python dependencies into the Pixi environment...

if not defined CUDA_PATH (
    set "CUDA_PATH=D:\cuda"
)

echo [pixi] Using CUDA_PATH=%CUDA_PATH%

python -m pip install --upgrade pip wheel
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip/wheel.
    exit /b 1
)

echo [pixi] Installing PyTorch with CUDA 12.8 wheels...
python -m pip install --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch CUDA wheels.
    exit /b 1
)

set "GGML_CUDA=1"
set "CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CXX_FLAGS=/Zc:preprocessor -DCMAKE_CUDA_FLAGS=-Xcompiler=/Zc:preprocessor"

echo [pixi] Installing llama-cpp-python with CUDA support...
python -m pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
if errorlevel 1 (
    echo ERROR: Failed to install llama-cpp-python.
    echo Make sure CUDA Toolkit and CMake are installed.
    exit /b 1
)

echo [pixi] Installing pywhispercpp with CUDA support...
python -m pip install --upgrade --force-reinstall pywhispercpp --no-cache-dir
if errorlevel 1 (
    echo ERROR: Failed to install pywhispercpp.
    exit /b 1
)

echo [pixi] Installing remaining project requirements...
python -m pip uninstall -y PySide6 PySide6_Addons PySide6_Essentials shiboken6 >nul 2>&1
python -m pip install --upgrade -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.txt dependencies.
    exit /b 1
)

echo [pixi] Enforcing setuptools compatibility for ROS colcon builds...
python tools\ensure_setuptools_compat.py
if errorlevel 1 (
    echo ERROR: Failed to install compatible setuptools.
    exit /b 1
)

echo [pixi] Python dependency bootstrap complete.
