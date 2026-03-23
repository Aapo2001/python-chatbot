#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-$(pwd)/config.json}"
VOICE_LOOP="${VOICE_LOOP:-false}"
TTS_PLAYBACK="${TTS_PLAYBACK:-true}"

if [[ ! -f "install/setup.bash" ]]; then
    echo "ERROR: install/setup.bash not found. Run \`pixi run build\` first." >&2
    exit 1
fi

source install/setup.bash
ros2 launch voice_chatbot_ros voice_chatbot.launch.py \
    config_path:="${CONFIG_PATH}" \
    enable_voice_loop:="${VOICE_LOOP}" \
    enable_tts_playback:="${TTS_PLAYBACK}"
