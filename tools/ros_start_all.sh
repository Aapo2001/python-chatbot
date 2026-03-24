#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  ros_start_all.sh — Launch all ROS 2 voice chatbot processes
#
#  Opens four tabs in a single GNOME Terminal window:
#    Tab 1: STT node   (mic + VAD + Whisper)
#    Tab 2: LLM node   (LLaMA/GGUF chat inference)
#    Tab 3: TTS node   (Coqui TTS + speaker playback)
#    Tab 4: GUI        (PySide6, delayed 2 s to let nodes start)
#
#  Requires gnome-terminal.  Falls back to manual instructions.
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."
WD="$(pwd)"

echo "Starting ROS 2 voice chatbot..."

# Build the ROS 2 workspace first (colcon build).
pixi run build

if command -v gnome-terminal &>/dev/null; then
    # Launch all four as tabs in one gnome-terminal window.
    # The GUI tab sleeps 2 seconds so the ROS nodes can advertise
    # their topics before the GUI tries to subscribe.
    gnome-terminal --window \
        --tab --title="STT" -- bash -c "cd '$WD' && pixi run ros-stt; exec bash" \
        --tab --title="LLM" -- bash -c "cd '$WD' && pixi run ros-llm; exec bash" \
        --tab --title="TTS" -- bash -c "cd '$WD' && pixi run ros-tts; exec bash" \
        --tab --title="GUI" -- bash -c "cd '$WD' && sleep 2 && pixi run ros-app; exec bash"
else
    echo "gnome-terminal not found. Run each manually in separate terminals:"
    echo "  pixi run ros-stt"
    echo "  pixi run ros-llm"
    echo "  pixi run ros-tts"
    echo "  pixi run ros-app"
    exit 1
fi

echo "All tabs launched."
