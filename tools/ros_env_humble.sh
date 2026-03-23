#!/usr/bin/env bash
set -euo pipefail

ROS_UNDERLAY="/opt/ros/humble/setup.bash"

if [[ ! -f "${ROS_UNDERLAY}" ]]; then
    echo "ERROR: ROS 2 Humble underlay not found at ${ROS_UNDERLAY}" >&2
    echo "Install ROS 2 Humble first and make sure /opt/ros/humble exists." >&2
    exit 1
fi

source "${ROS_UNDERLAY}"

if [[ -f "install/setup.bash" ]]; then
    source "install/setup.bash"
fi
