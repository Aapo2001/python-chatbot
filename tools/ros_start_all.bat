@echo off
REM ─────────────────────────────────────────────────────────────────
REM  ros_start_all.bat — Launch all ROS 2 voice chatbot processes
REM
REM  Opens four tabs in a single Windows Terminal window:
REM    Tab 1: STT node   (mic + VAD + Whisper)
REM    Tab 2: LLM node   (LLaMA/GGUF chat inference)
REM    Tab 3: TTS node   (Coqui TTS + speaker playback)
REM    Tab 4: GUI        (PySide6, delayed ~2 s to let nodes start)
REM
REM  Requires Windows Terminal (`wt`) on PATH (default on Win 11).
REM ─────────────────────────────────────────────────────────────────
setlocal

echo Starting ROS 2 voice chatbot...

REM Build the ROS 2 workspace first (colcon build).
cd /d "%~dp0.."
pixi run build
if errorlevel 1 (
    echo ERROR: Build failed.
    exit /b 1
)

REM Launch all 4 as tabs in a single Windows Terminal window.
REM `ping -n 3` provides a ~2-second delay for the GUI tab so the
REM ROS nodes have time to advertise their topics before the GUI
REM tries to subscribe.
wt -w 0 new-tab --title "STT" -d "%CD%" cmd /k "pixi run ros-stt" ; ^
   new-tab --title "LLM" -d "%CD%" cmd /k "pixi run ros-llm" ; ^
   new-tab --title "TTS" -d "%CD%" cmd /k "pixi run ros-tts" ; ^
   new-tab --title "GUI" -d "%CD%" cmd /k "ping -n 3 127.0.0.1 >nul && pixi run ros-app"

echo All tabs launched.
