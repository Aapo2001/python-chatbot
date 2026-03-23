#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/ros_env_humble.sh"

colcon build --packages-select voice_chatbot_ros --symlink-install
