# Voice Chatbot

Local speech-to-speech voice assistant with PySide6 GUI, CLI, and ROS 2 Humble integration.

## Pipeline

```
Microphone → AudioIO → VoiceActivityDetector → SpeechToText → ChatLLM → TextToSpeech → Speaker
               ▲                                                                          │
               └── clear_queue() + vad.reset() after playback (prevents self-triggering) ─┘
```

## Quick Start

```bash
# 1. First-time setup
pixi install                        # create pixi environment
pixi run install-python-deps        # install pip packages (torch, TTS, etc.)
pixi run setup-models               # download model files (LLM, Whisper, VAD, TTS)

# 2. Run the GUI (primary interface)
pixi run app
# or: python app.py

# 3. Run the CLI (headless)
python chatbot.py
```

## ROS 2 (via pixi + robostack)

```bash
pixi run build                  # colcon build (once)

# Option A — all four tabs in one Windows Terminal window:
pixi run ros-start              # builds, then opens STT + LLM + TTS + GUI tabs

# Option B — split nodes in separate terminals (build once first):
pixi run ros-stt                # STT node: mic + VAD + Whisper
pixi run ros-llm                # LLM node: LLaMA chat inference
pixi run ros-tts                # TTS node: Coqui TTS + audio playback
pixi run ros-app                # PySide6 GUI (connects to running nodes)

# Option C — launch file (all 3 nodes in one process group):
pixi run ros-launch

# Legacy monolithic node (loads everything sequentially):
pixi run ros-run
```

## Project Structure

### Entry points

| File | Purpose |
|------|---------|
| `app.py` | PySide6 desktop GUI — standalone (loads all models in-process) |
| `ros_app.py` | PySide6 desktop GUI — ROS 2 (connects to split nodes via topics) |
| `chatbot.py` | Headless CLI runner (single-threaded audio loop) |
| `setup_models.py` | Downloads and validates all models before first run |

### Pipeline modules

| File | Purpose |
|------|---------|
| `config.py` | `Config` dataclass + JSON persistence (`config.json`) |
| `audio_io.py` | Microphone capture + playback via `sounddevice` |
| `vad.py` | Silero-VAD wrapper with pre-buffer and energy gating |
| `stt.py` | Whisper STT via `faster-whisper` (CTranslate2 backend) |
| `llm.py` | LLaMA multi-turn chat via `llama-cpp-python` (GGUF) |
| `tts_engine.py` | Coqui TTS wrapper (local model or model zoo) |

### ROS 2 nodes

| File | Purpose |
|------|---------|
| `voice_chatbot_ros/stt_node.py` | STT split node (mic + VAD + Whisper) |
| `voice_chatbot_ros/llm_node.py` | LLM split node (chat inference) |
| `voice_chatbot_ros/tts_node.py` | TTS split node (synthesis + playback) |
| `voice_chatbot_ros/node.py` | Monolithic node (legacy, all-in-one) |
| `launch/voice_chatbot.launch.py` | Launch file for all 3 split nodes |

### Build and tooling

| File | Purpose |
|------|---------|
| `pixi.toml` | Pixi manifest (deps, channels, tasks) |
| `setup.py` | ROS 2 package setup (colcon / ament_python) |
| `config.json` | Runtime configuration (saved by GUI) |
| `tools/ros_start_all.bat` / `.sh` | Launch all 4 tabs (STT, LLM, TTS, GUI) |
| `tools/ros_run_node_pixi.bat` / `.sh` | Run a single ROS 2 node via pixi |
| `tools/ros_launch_pixi.bat` / `.sh` | Run the launch file via pixi |
| `tools/install_python_windows.bat` | Install pip deps (Windows, CUDA 12.8) |
| `tools/install_python_linux.sh` | Install pip deps (Linux, generic CUDA) |
| `tools/ensure_setuptools_compat.py` | Enforce setuptools 69.5–79.x for colcon |

## Key Libraries

| Library | Purpose | Notes |
|---------|---------|-------|
| **PySide6** | Qt 6 GUI | pip-installed; conflicts with Robostack Qt — see DLL workaround in `app.py` |
| **silero-vad** | Voice activity detection | Loaded via `torch.hub` |
| **faster-whisper** | CTranslate2-based Whisper STT | CUDA support for GPU inference |
| **llama-cpp-python** | LLaMA GGUF inference | CUDA offloading via `n_gpu_layers` |
| **coqui-tts** | Text-to-speech | Requires `espeak-ng` for phonemisation |
| **sounddevice** | Audio I/O | Callback-based input, blocking playback |
| **huggingface-hub** | Model downloads | Used by `setup_models.py` |

## Environment

- **Python**: 3.11 (managed by pixi)
- **CUDA**: Required for GPU inference. DLL path defaults to `D:\cuda` (override with `CUDA_PATH` env var)
- **espeak-ng**: Required for Coqui TTS phonemisation. Install to `C:\Program Files\eSpeak NG` on Windows.
- **Platforms**: Windows 11 (primary), Linux 64-bit
- **Package manager**: pixi (with `robostack-humble` + `conda-forge` channels)
- **Python deps**: pip-installed into pixi env via `pixi run install-python-deps`

## Configuration

All runtime settings are in `config.json`. The GUI reads/writes this file. Key fields:
- `language` — STT/TTS language code (default: `"fi"`)
- `whisper_model` — Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v*`)
- `llm_model_path` — Path to GGUF model file
- `tts_model` — Coqui TTS model identifier (or use local `tts_model_path` + `tts_config_path`)
- `vad_threshold`, `min_silence_duration_ms`, `vad_pre_buffer_ms` — VAD sensitivity
- `llm_temperature`, `llm_n_ctx`, `llm_max_tokens` — LLM generation control
- `max_conversation_turns` — history trimming limit

## Conventions

- **Finnish-first** — UI labels, status messages, and the default system prompt are in Finnish
- **CUDA DLL setup** — must run before any CUDA-dependent imports (torch, llama_cpp, faster_whisper)
- **Deferred imports** — heavy libraries are imported inside worker threads to keep GUI startup instant
- **Thread safety** — GUI worker uses `QThread` with `QueuedConnection` signals; ROS nodes use queue + daemon thread patterns
- **Self-trigger prevention** — after TTS playback, the audio queue is cleared and VAD is reset
- **Models directory** — `models/` (gitignored, created by `setup_models.py`)
