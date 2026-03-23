#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/ros_env_humble.sh"

ros2 run voice_chatbot_ros voice_chatbot_node --ros-args -p config_path:="${1:-$(pwd)/config.json}"
