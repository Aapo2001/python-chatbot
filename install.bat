@echo off
setlocal

echo ================================================
echo   Voice Chatbot - Installation Script
echo ================================================
echo.

set "PIXI_EXE="
call :resolve_pixi

if not defined PIXI_EXE (
    echo [1/3] Installing pixi...
    powershell -ExecutionPolicy Bypass -Command "irm -useb https://pixi.sh/install.ps1 | iex"
    if errorlevel 1 (
        echo ERROR: Failed to install pixi.
        pause
        exit /b 1
    )

    call :resolve_pixi
)

if not defined PIXI_EXE (
    echo ERROR: pixi was installed but could not be found in PATH.
    echo Try restarting the terminal and running this script again.
    pause
    exit /b 1
)

echo [2/3] Creating or updating the pixi environment...
"%PIXI_EXE%" install
if errorlevel 1 (
    echo ERROR: Failed to create the pixi environment.
    pause
    exit /b 1
)

echo [3/3] Installing Python and CUDA-backed dependencies via pixi...
"%PIXI_EXE%" run install-python-deps
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies through pixi.
    pause
    exit /b 1
)

echo.
echo ================================================
echo   Installation complete!
echo ================================================
echo.
echo Next steps:
echo   1. Download models:    pixi run setup-models
echo   2. Start chatbot:
echo        GUI mode:         pixi run app
echo        Terminal mode:    pixi run chatbot
echo   3. ROS 2 Humble (optional):
echo        Build package:    pixi run build
echo        Run node:         pixi run ros-run
echo   4. Optional shell:     pixi shell
echo.
pause
exit /b 0

:resolve_pixi
where pixi >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%I in ('where pixi') do (
        if not defined PIXI_EXE set "PIXI_EXE=%%I"
    )
)

if not defined PIXI_EXE if exist "%USERPROFILE%\.pixi\bin\pixi.exe" (
    set "PIXI_EXE=%USERPROFILE%\.pixi\bin\pixi.exe"
)

if not defined PIXI_EXE if exist "%LocalAppData%\pixi\bin\pixi.exe" (
    set "PIXI_EXE=%LocalAppData%\pixi\bin\pixi.exe"
)

goto :eof
