# ROS 2 Humble Support

## Overview

This repository now includes a ROS 2 Python package named `voice_chatbot_ros`. The ROS node reuses the existing local chatbot modules and exposes them through ROS topics and a service.

The node supports two operating patterns:

- text mode: subscribe to incoming user text, generate a response, and optionally speak it locally
- voice mode: run the existing microphone -> VAD -> STT pipeline locally, publish transcripts, generate a response, and optionally play TTS audio locally

## ROS Interfaces

Topics:

- `~/user_text` (`std_msgs/msg/String`, subscription)
  Send user text into the chatbot.
- `~/assistant_text` (`std_msgs/msg/String`, publisher)
  Publishes assistant replies.
- `~/transcript` (`std_msgs/msg/String`, publisher)
  Publishes speech-to-text output when voice mode is enabled.
- `~/status` (`std_msgs/msg/String`, publisher)
  Publishes coarse runtime state such as `initializing`, `ready`, `speech_detected`, `transcribing`, `llm_responding`, `speaking`, and `error`.
- `~/log` (`std_msgs/msg/String`, publisher)
  Publishes operator-facing log messages.

Service:

- `~/clear_history` (`std_srvs/srv/Trigger`)
  Clears the in-memory LLM conversation history.

## Parameters

- `config_path` (`string`, default: `config.json`)
  Path passed to `Config.load()`.
- `load_config_file` (`bool`, default: `true`)
  If `false`, the node uses the `Config` dataclass defaults.
- `enable_voice_loop` (`bool`, default: `false`)
  Enables microphone capture plus VAD and STT.
- `enable_tts_playback` (`bool`, default: `true`)
  Enables local TTS synthesis and playback for assistant replies.

## Build

The supported Pixi workflow for ROS 2 Humble assumes Ubuntu 22.04 or another Linux system with a standard ROS underlay at `/opt/ros/humble`.

Install the Python and ML stack through Pixi:

```bash
pixi install
pixi run install-python-deps
```

Then build the ROS package through the Pixi ROS task:

```bash
pixi run ros-build
```

ROS-side dependencies are declared in `package.xml`:

- `rclpy`
- `std_msgs`
- `std_srvs`
- `launch`
- `launch_ros`

The Linux ROS tasks source `/opt/ros/humble/setup.bash` automatically and then source the local workspace overlay from `install/setup.bash` when it exists.

## Run

Run the node in text mode:

```bash
pixi run ros-run /absolute/path/to/config.json
```

Launch with the included launch file:

```bash
VOICE_LOOP=true pixi run ros-launch /absolute/path/to/config.json
```

To keep TTS playback enabled explicitly:

```bash
VOICE_LOOP=true TTS_PLAYBACK=true pixi run ros-launch /absolute/path/to/config.json
```

## Quick Smoke Test

Publish a text request:

```bash
ros2 topic pub /voice_chatbot/user_text std_msgs/msg/String "{data: 'Hei, kuka olet?'}" --once
```

Watch the response and status topics:

```bash
ros2 topic echo /voice_chatbot/assistant_text
ros2 topic echo /voice_chatbot/status
```

Clear chat history:

```bash
ros2 service call /voice_chatbot/clear_history std_srvs/srv/Trigger "{}"
```

## Notes

- The ROS package is additive. Existing `app.py`, `chatbot.py`, and `setup_models.py` workflows remain unchanged.
- The node serializes requests through a worker thread so the underlying LLM wrapper is used from one place at a time.
- In voice mode, the node clears queued microphone audio and resets VAD after TTS playback to reduce self-triggering.
- The same config caveats still apply: the default project configuration is Finnish-first, and model files must already exist at the configured paths.
- `pixi run ros-build`, `pixi run ros-run`, and `pixi run ros-launch` are Linux/Humble-oriented tasks. The Windows `install.bat` flow remains focused on the desktop app.
