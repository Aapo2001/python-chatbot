@echo off
setlocal

set "CONFIG_PATH=%~1"
if not defined CONFIG_PATH (
    set "CONFIG_PATH=%CD%\config.json"
)

if not exist "install\setup.bat" (
    echo ERROR: install\setup.bat not found. Run `pixi run build` first.
    exit /b 1
)

call "install\setup.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate the local ROS workspace from install\setup.bat.
    exit /b 1
)

ros2 launch voice_chatbot_ros voice_chatbot.launch.py ^
    config_path:="%CONFIG_PATH%"
