# ROS 2 Humble Support

## Overview

The `voice_chatbot_ros` package provides ROS 2 integration for the local voice chatbot pipeline. It offers two architectures:

### Split-node architecture (recommended)

Three independent nodes, each loading only the models they need. Because they start in separate processes, models load **in parallel** and the system is ready much faster than the monolithic node.

```
┌─────────────┐     /user_text      ┌─────────────┐    /assistant_text    ┌─────────────┐
│   STT node  │ ──────────────────► │   LLM node  │ ──────────────────► │   TTS node  │
│ mic+VAD+STT │     /transcript     │  LLaMA/GGUF │                      │ Coqui + play│
└──────▲──────┘                      └─────────────┘                      └──────┬──────┘
       │                                                                          │
       └──────────────────── /tts_done ◄──────────────────────────────────────────┘
                          (VAD reset signal)
```

| Node | File | Models loaded |
|------|------|--------------|
| `voice_stt` | `voice_chatbot_ros/stt_node.py` | AudioIO, Silero-VAD, faster-whisper |
| `voice_llm` | `voice_chatbot_ros/llm_node.py` | llama-cpp-python (GGUF) |
| `voice_tts` | `voice_chatbot_ros/tts_node.py` | Coqui TTS, AudioIO (playback only) |

### Monolithic node (legacy)

A single node (`voice_chatbot_ros/node.py`) that loads everything sequentially. Supports text-only mode (subscribe to `~/user_text`) and voice mode (microphone → VAD → STT). Kept for backwards compatibility but the split architecture is preferred.

## ROS Interfaces

### Split nodes (in `/voice_chatbot` namespace)

**Topics:**

| Topic | Type | Direction | Publisher → Subscriber |
|-------|------|-----------|----------------------|
| `user_text` | `std_msgs/String` | STT → LLM | Transcribed speech or typed text |
| `assistant_text` | `std_msgs/String` | LLM → TTS | LLM reply to be spoken |
| `transcript` | `std_msgs/String` | STT → GUI | Copy of transcription for display |
| `tts_done` | `std_msgs/String` | TTS → STT | Signal to reset VAD after playback |
| `status` | `std_msgs/String` | All → GUI | Pipeline state changes |
| `log` | `std_msgs/String` | All → GUI | Human-readable log messages |

**Status values:** `initializing`, `listening`, `speech_detected`, `transcribing`, `llm_responding`, `speaking`, `ready`, `error`

**Service:**

| Service | Type | Node | Description |
|---------|------|------|-------------|
| `clear_history` | `std_srvs/Trigger` | LLM | Erase conversation memory |

### Monolithic node (private `~/` topics)

Same topics and service but using ROS 2 private names (`~/user_text`, `~/assistant_text`, etc.).

## Parameters

All nodes share these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_path` | string | `config.json` | Path to the configuration file |
| `load_config_file` | bool | `true` | If false, use `Config` dataclass defaults |

Monolithic node only:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_voice_loop` | bool | `false` | Enable microphone + VAD + STT |
| `enable_tts_playback` | bool | `true` | Enable TTS synthesis and speaker output |

## Build

ROS 2 Humble is installed via the `robostack-humble` conda channel into the pixi environment — no system-level ROS installation required.

### Initial setup

```bash
pixi install                      # create the environment
pixi run install-python-deps      # install pip packages (torch, TTS, etc.)
pixi run setup-models             # download model files
```

### Build the ROS package

```bash
pixi run build                    # colcon build --symlink-install
```

The build task automatically checks that `setuptools` is in the 69.5–79.x range required by colcon's `setup.py develop` path.

### Package dependencies (`package.xml`)

- `rclpy`, `std_msgs`, `std_srvs`, `launch`, `launch_ros`

## Run

### Split nodes (recommended)

**One command — all four tabs in Windows Terminal:**

```bash
pixi run ros-start                # builds, then opens 4 tabs: STT, LLM, TTS, GUI
```

**Or manually in separate terminals (build once first):**

```bash
# Terminal 1 — build
pixi run build

# Terminal 2 — STT node
pixi run ros-stt

# Terminal 3 — LLM node
pixi run ros-llm

# Terminal 4 — TTS node
pixi run ros-tts

# Terminal 5 — GUI (optional)
pixi run ros-app
```

**Launch file (all 3 nodes in one process group):**

```bash
pixi run ros-launch
```

### Monolithic node (legacy)

```bash
pixi run ros-run
```

## Quick Smoke Test

Publish a text request:

```bash
ros2 topic pub /voice_chatbot/user_text std_msgs/msg/String "{data: 'Hei, kuka olet?'}" --once
```

Watch responses and status:

```bash
ros2 topic echo /voice_chatbot/assistant_text
ros2 topic echo /voice_chatbot/status
```

Clear chat history:

```bash
ros2 service call /voice_chatbot/clear_history std_srvs/srv/Trigger "{}"
```

## Architecture Notes

- **Parallel model loading** — The split-node architecture loads VAD + Whisper, LLaMA, and Coqui TTS in three separate processes simultaneously, cutting startup time roughly to the duration of the slowest model load.
- **Self-trigger prevention** — After TTS playback, the TTS node publishes `tts_done`. The STT node subscribes to this signal and clears its microphone buffer + resets VAD state, so the assistant's own voice is not mistaken for user speech.
- **Request serialisation** — Both the LLM and TTS nodes use a queue + worker thread pattern so that concurrent messages are processed in order without racing.
- **Shared config** — All nodes load the same `config.json` at startup. GUI settings changes require node restart to take effect.
- **Finnish-first** — Default language, UI labels, and system prompt are in Finnish. Change the `language` field in `config.json` (or the GUI) to switch.
- **Pixi-native** — All build and run tasks work through pixi on both Windows and Linux, with platform-specific scripts under `tools/`.
