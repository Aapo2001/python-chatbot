@echo off
REM ─────────────────────────────────────────────────────────────────
REM  ros_run_node_pixi.bat — Run a single ROS 2 voice chatbot node
REM
REM  Usage: ros_run_node_pixi.bat <node_executable_name> [config_path]
REM    e.g.: ros_run_node_pixi.bat voice_stt_node
REM           ros_run_node_pixi.bat voice_llm_node D:\my\config.json
REM
REM  Sources the local colcon workspace overlay (install\setup.bat)
REM  then starts the node in the /voice_chatbot namespace.
REM ─────────────────────────────────────────────────────────────────
setlocal

set "NODE_NAME=%~1"
if not defined NODE_NAME (
    echo ERROR: Usage: ros_run_node_pixi.bat ^<node_executable_name^>
    echo   e.g.: ros_run_node_pixi.bat voice_stt_node
    exit /b 1
)

set "CONFIG_PATH=%~2"
if not defined CONFIG_PATH (
    set "CONFIG_PATH=%CD%\config.json"
)

REM Verify the workspace has been built.
if not exist "install\setup.bat" (
    echo ERROR: install\setup.bat not found. Run `pixi run build` first.
    exit /b 1
)

REM Source the colcon workspace overlay.
call "install\setup.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate the local ROS workspace from install\setup.bat.
    exit /b 1
)

REM Start the node in the /voice_chatbot namespace with the config path.
ros2 run voice_chatbot_ros %NODE_NAME% --ros-args -p config_path:="%CONFIG_PATH%" -r __ns:=/voice_chatbot
