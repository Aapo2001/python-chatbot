#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  ros_run_node_pixi.sh — Run a single ROS 2 voice chatbot node
#
#  Usage: ros_run_node_pixi.sh <node_executable_name> [config_path]
#    e.g.: ros_run_node_pixi.sh voice_stt_node
#           ros_run_node_pixi.sh voice_llm_node /absolute/path/config.json
#
#  Sources the local colcon workspace overlay (install/setup.bash)
#  then starts the node in the /voice_chatbot namespace.
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

NODE_NAME="${1:?Usage: ros_run_node_pixi.sh <node_executable_name>}"
CONFIG_PATH="${2:-$(pwd)/config.json}"

# Verify the workspace has been built.
if [[ ! -f "install/setup.bash" ]]; then
    echo "ERROR: install/setup.bash not found. Run \`pixi run build\` first." >&2
    exit 1
fi

# Source the colcon workspace overlay and start the node.
source install/setup.bash
ros2 run voice_chatbot_ros "${NODE_NAME}" --ros-args -p config_path:="${CONFIG_PATH}" -r __ns:=/voice_chatbot
