#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/ros_env_humble.sh"

CONFIG_PATH="${1:-$(pwd)/config.json}"
VOICE_LOOP="${VOICE_LOOP:-false}"
TTS_PLAYBACK="${TTS_PLAYBACK:-true}"

ros2 launch voice_chatbot_ros voice_chatbot.launch.py \
    config_path:="${CONFIG_PATH}" \
    enable_voice_loop:="${VOICE_LOOP}" \
    enable_tts_playback:="${TTS_PLAYBACK}"
