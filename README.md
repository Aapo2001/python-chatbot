# Voice Chatbot

Local voice chatbot for Windows with a Finnish-first default configuration. The application captures microphone audio, detects speech with Silero VAD, transcribes it with Whisper, generates a reply with a local GGUF LLM, and speaks the reply with Coqui TTS.

The repository contains two entry points:

- `app.py`: PySide6 desktop UI for configuring and running the chatbot.
- `chatbot.py`: terminal-only runner with the same audio pipeline.
- `voice_chatbot_ros/node.py`: ROS 2 Humble node that exposes the pipeline through ROS topics and a service.

## What The Code Does

Runtime flow:

1. `AudioIO` captures 16 kHz mono microphone audio in fixed-size chunks.
2. `VoiceActivityDetector` buffers audio until Silero VAD reports speech start and end.
3. `SpeechToText` transcribes the captured utterance with `pywhispercpp`.
4. `ChatLLM` sends the user text plus recent conversation history to `llama-cpp-python`.
5. `TextToSpeech` synthesizes the assistant reply with Coqui TTS.
6. `AudioIO` plays the generated speech back through the default output device.

The GUI wraps this pipeline in a background `QThread` and exposes model and runtime settings in a sidebar.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `app.py` | Main desktop application and worker thread orchestration |
| `chatbot.py` | Console entry point |
| `config.py` | `Config` dataclass and JSON load/save helpers |
| `audio_io.py` | Microphone capture, playback, and queue handling |
| `vad.py` | Silero VAD integration plus pre-buffer logic |
| `stt.py` | Whisper STT wrapper |
| `llm.py` | `llama-cpp-python` chat wrapper and history management |
| `tts_engine.py` | Coqui TTS wrapper |
| `setup_models.py` | One-time model download and validation script |
| `install.bat` | Windows setup script for Python packages and CUDA builds |
| `pixi.toml` | Pixi workspace manifest for the base toolchain and common tasks |
| `pixi.lock` | Resolved Pixi lockfile for the base environment |
| `tools/install_python_windows.bat` | Pixi bootstrap script for CUDA-enabled Python packages |
| `config.json` | Persisted GUI configuration |
| `voice_chatbot_ros/` | ROS 2 Humble package and node implementation |
| `launch/` | ROS 2 launch file |
| `package.xml`, `setup.py`, `setup.cfg` | ROS 2 `ament_python` package metadata |

More implementation detail is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
ROS-specific usage is in [docs/ROS2.md](docs/ROS2.md).

## Environment Assumptions

The codebase is currently optimized for a Windows workstation with local GPU inference:

- `app.py` and `chatbot.py` add CUDA DLL paths from `CUDA_PATH` or `D:\cuda`.
- `install.bat` installs CUDA-enabled PyTorch, `llama-cpp-python`, and `pywhispercpp`.
- The default LLM is a GGUF file under `models/`.
- The default voice assistant language is Finnish (`fi`).

The code can fall back to CPU for some components, but the intended deployment is local GPU acceleration.

## Installation

The recommended setup path is now Pixi. The repository ships with a `pixi.toml` workspace manifest plus a Windows bootstrap script that installs the CUDA-specific Python packages inside the Pixi environment.

### 1. System prerequisites

- Windows
- `pixi` or permission to let `install.bat` install it
- CUDA Toolkit if you want GPU acceleration for PyTorch and `llama-cpp-python`
- CMake and a working build toolchain for Python packages with native extensions
- Microphone and speakers/headphones configured as default audio devices
- For ROS 2 Humble on Linux: a system ROS install under `/opt/ros/humble`

### 2. Create the Pixi environment and install packages

Run:

```powershell
install.bat
```

What the script does:

- installs `pixi` if it is missing
- creates or updates the local Pixi environment from `pixi.toml`
- runs the Pixi bootstrap task that installs the Python packages needed by the project

The Pixi workspace provides:

- Python 3.10
- `pip`
- `cmake`
- `git`
- `ninja`

The bootstrap task installs:

- CUDA-enabled `torch`, `torchvision`, `torchaudio`
- `llama-cpp-python` compiled with `GGML_CUDA=on`
- `pywhispercpp` compiled with CUDA flags
- the remaining packages from `requirements.txt`

`requirements.txt` is still not the full environment by itself. `torch`, `llama-cpp-python`, and `pywhispercpp` are installed separately because they need a custom wheel index or CUDA-specific build flags.

### 3. Direct Pixi workflow

If you do not want to use `install.bat`, the equivalent commands are:

```powershell
pixi install
pixi run install-python-deps
```

### 4. Download models

Run:

```powershell
pixi run setup-models
```

This script:

- checks CUDA visibility in PyTorch
- initializes the Silero VAD model
- downloads or validates the configured Whisper model
- downloads the configured GGUF LLM from Hugging Face if missing
- initializes the configured Coqui TTS model

## Running The Application

Desktop UI:

```powershell
pixi run app
```

Terminal mode:

```powershell
pixi run chatbot
```

ROS 2 Humble node:

```bash
pixi run ros-run /absolute/path/to/config.json
```

This Pixi ROS workflow is intended for Linux systems with ROS 2 Humble installed under `/opt/ros/humble`.

## Configuration

Configuration is defined in `config.py` and can be persisted to `config.json`.

Important settings:

- Audio: `sample_rate`, `channels`, `chunk_samples`
- VAD: `vad_threshold`, `min_silence_duration_ms`, `speech_pad_ms`, `min_speech_duration_ms`, `vad_pre_buffer_ms`
- STT: `language`, `whisper_model`, `whisper_n_threads`
- LLM: `llm_model_path`, `llm_n_gpu_layers`, `llm_n_ctx`, `llm_max_tokens`, `llm_temperature`, `llm_system_prompt`, `max_conversation_turns`
- TTS: `tts_model`, `tts_gpu`
- Download metadata: `llm_repo_id`, `llm_filename`

### Configuration behavior to know

- The GUI loads from `config.json` through `Config.load()` and writes the current sidebar values back to disk when you start the worker.
- `chatbot.py` currently constructs `Config()` directly, so it uses dataclass defaults instead of loading `config.json`.
- `setup_models.py` also uses `Config()` directly, so it validates or downloads models based on default values unless the script is changed.

## GUI Behavior

The desktop app provides:

- a settings sidebar for language, Whisper, LLM, TTS, and VAD parameters
- a chat panel for user and assistant messages
- a system log panel that receives redirected `stdout` and `stderr`
- start, stop, restart, and clear-chat controls

Behavior details from the current code:

- Settings remain editable while the worker is running.
- Changing settings during runtime requires `Käynnistä uudelleen` to rebuild the worker with the new values.
- `Tyhjennä keskustelu` clears the visible chat panel only. It does not clear the LLM conversation history; history resets only when a new `ChatLLM` instance is created, such as after restart.

## Architecture Notes

- The VAD implementation keeps a rolling pre-buffer so the first syllables are not clipped before speech onset is confirmed.
- After TTS playback, the app clears queued microphone chunks and resets VAD state to reduce the chance of transcribing its own synthesized output.
- The LLM wrapper stores alternating user and assistant messages and trims the oldest turns once `max_conversation_turns` is exceeded.
- The GUI runs model loading and the audio loop in `ChatbotWorker`, a `QThread`, so the main UI thread stays responsive.

## Operational Limitations

- There are no automated tests in the repository.
- Audio device selection is not exposed; capture and playback use the default system devices through `sounddevice`.
- The code and UI text are partly Finnish and partly English.
- Model initialization happens synchronously inside the worker or CLI startup path, so startup cost depends on model size.
- ROS 2 support assumes the ROS Python packages and the ML/audio dependencies are available in the same runtime environment.
- The Pixi ROS tasks assume a Linux ROS 2 Humble underlay at `/opt/ros/humble/setup.bash`.

## Suggested First Run

1. Run `install.bat`.
2. Run `pixi run setup-models`.
3. Start `pixi run app`.
4. Confirm the GGUF model path in the left sidebar.
5. Click `Käynnistä` and watch the system log for CUDA and model load status.
