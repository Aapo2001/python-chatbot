@echo off
echo ================================================
echo   Voice Chatbot - Installation Script
echo ================================================
echo.

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10 or 3.11 from https://www.python.org/
    pause
    exit /b 1
)

REM Create virtual environment
echo [1/6] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo   Virtual environment created.
) else (
    echo   Virtual environment already exists.
)

REM Activate virtual environment
echo [2/6] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install PyTorch with CUDA
echo [3/6] Installing PyTorch with CUDA 12.8 support (RTX 50 series)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch.
    pause
    exit /b 1
)

REM Install llama-cpp-python with CUDA
echo [4/6] Installing llama-cpp-python with CUDA support...
set "CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CXX_FLAGS=/Zc:preprocessor -DCMAKE_CUDA_FLAGS=-Xcompiler=/Zc:preprocessor"
pip install llama-cpp-python --no-cache-dir
if errorlevel 1 (
    echo ERROR: Failed to install llama-cpp-python.
    echo Make sure CUDA Toolkit and CMake are installed.
    echo   CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
    echo   CMake: https://cmake.org/download/
    pause
    exit /b 1
)

REM Install pywhispercpp with CUDA
echo [5/6] Installing pywhispercpp with CUDA support...
set GGML_CUDA=1
set "CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CXX_FLAGS=/Zc:preprocessor -DCMAKE_CUDA_FLAGS=-Xcompiler=/Zc:preprocessor"
pip install pywhispercpp --no-cache-dir
if errorlevel 1 (
    echo ERROR: Failed to install pywhispercpp.
    pause
    exit /b 1
)

REM Install remaining dependencies
echo [6/6] Installing remaining dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ================================================
echo   Installation complete!
echo ================================================
echo.
echo Next steps:
echo   1. Activate the venv:  venv\Scripts\activate.bat
echo   2. Download models:    python setup_models.py
echo   3. Start chatbot:      python chatbot.py
echo.
pause
