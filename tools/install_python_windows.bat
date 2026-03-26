@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Activate MSVC developer environment if cl.exe is not on PATH.
REM This MUST happen before setlocal so the environment propagates.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"

where cl.exe >nul 2>&1
if errorlevel 1 (
    if not exist "%VSWHERE%" (
        echo ERROR: Visual Studio Build Tools were not found.
        echo Install Visual Studio 2022 Build Tools with the Desktop development with C++ workload.
        exit /b 1
    )

    set "VS_INSTALL_PATH="
    for /f "usebackq delims=" %%i in (`""!VSWHERE!" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath"`) do set "VS_INSTALL_PATH=%%i"

    if not defined VS_INSTALL_PATH (
        echo ERROR: No Visual Studio installation with MSVC C++ tools was found.
        echo Install Visual Studio 2022 Build Tools with the Desktop development with C++ workload.
        exit /b 1
    )

    call "!VS_INSTALL_PATH!\VC\Auxiliary\Build\vcvarsall.bat" x64
    if errorlevel 1 (
        echo ERROR: Failed to initialize the MSVC build environment.
        exit /b 1
    )
)

echo [pixi] Installing Python dependencies into the Pixi environment...

if not defined CUDA_PATH (
    set "CUDA_PATH=D:\cuda"
)

echo [pixi] Using CUDA_PATH=%CUDA_PATH%

where cl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: cl.exe is not available on PATH.
    echo Install Visual Studio 2022 Build Tools with the Desktop development with C++ workload.
    exit /b 1
)

where cmake.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: cmake.exe is not available on PATH.
    exit /b 1
)

where ninja.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: ninja.exe is not available on PATH.
    exit /b 1
)

if exist "%CUDA_PATH%\bin\nvcc.exe" (
    set "CUDACXX=%CUDA_PATH%\bin\nvcc.exe"
) else (
    echo ERROR: nvcc.exe was not found under CUDA_PATH.
    echo Set CUDA_PATH to your CUDA Toolkit install directory, for example C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
    exit /b 1
)

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
set "CC=cl.exe"
set "CXX=cl.exe"
set "CMAKE_ARGS=-G Ninja -DGGML_CUDA=on -DCMAKE_CXX_FLAGS=/Zc:preprocessor -DCMAKE_CUDA_FLAGS=-Xcompiler=/Zc:preprocessor"

echo [pixi] Installing llama-cpp-python with CUDA support...
python -m pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
if errorlevel 1 (
    echo ERROR: Failed to install llama-cpp-python.
    echo Make sure CUDA Toolkit and CMake are installed.
    exit /b 1
)

echo [pixi] Installing faster-whisper...
python -m pip install --upgrade --force-reinstall faster-whisper --no-cache-dir
if errorlevel 1 (
    echo ERROR: Failed to install faster-whisper.
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
