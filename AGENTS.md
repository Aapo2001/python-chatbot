# Voice Chatbot

Local speech-to-speech voice assistant with PySide6 GUI and CLI. Available on PyPI: `pip install voice-chatbot[all]`

## Pipeline

```
Microphone → AudioIO → VoiceActivityDetector → SpeechToText → ChatLLM → TextToSpeech → Speaker
               ▲                                                                          │
               └── clear_queue() + vad.reset() after playback (prevents self-triggering) ─┘
```

## Quick Start

### Install from PyPI

```bash
pip install voice-chatbot[all]
voice-chatbot-setup-models
voice-chatbot-app
```

### Install from source (with pixi)

```bash
pixi install
pixi run install-python-deps
pixi run setup-models
pixi run app
```

## Project Structure

### Entry points

| File | Purpose |
|------|---------|
| `voice_chatbot/app.py` | PySide6 desktop GUI (loads all models in-process) |
| `voice_chatbot/chatbot.py` | Headless CLI runner (single-threaded audio loop) |
| `voice_chatbot/setup_models.py` | Downloads and validates all models before first run |

### Pipeline modules

| File | Purpose |
|------|---------|
| `voice_chatbot/config.py` | `Config` dataclass + JSON persistence (`config.json`) |
| `voice_chatbot/audio_io.py` | Microphone capture + playback via `sounddevice` |
| `voice_chatbot/vad.py` | Silero-VAD wrapper with pre-buffer and energy gating |
| `voice_chatbot/stt.py` | Whisper STT via `faster-whisper` (CTranslate2 backend) |
| `voice_chatbot/llm.py` | LLaMA multi-turn chat via `llama-cpp-python` (GGUF) |
| `voice_chatbot/tts_engine.py` | Coqui TTS wrapper (local model or model zoo) |
| `voice_chatbot/ui_common.py` | Shared PySide6 UI components (settings sidebar) |
| `voice_chatbot/platform_setup.py` | CUDA / PySide6 DLL setup |

### Build and tooling

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata + build config (setuptools backend) |
| `pixi.toml` | Pixi workspace manifest (deps, tasks) |
| `config.json` | Runtime configuration (saved by GUI) |
| `tools/install_python_windows.bat` | Install pip deps (Windows, CUDA 12.8) |
| `tools/install_python_linux.sh` | Install pip deps (Linux, generic CUDA) |

## Key Libraries

| Library | Purpose | Notes |
|---------|---------|-------|
| **PySide6** | Qt 6 GUI | pip-installed |
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
- **Package manager**: pixi (with `conda-forge` channel)
- **Python deps**: pip-installed into pixi env via `pixi run install-python-deps`

## Configuration

All runtime settings are in `config.json`. The GUI reads/writes this file. Key fields:
- `language` — STT/TTS language code (default: `"fi"`)
- `whisper_model` — Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`)
- `llm_model_path` — Path to GGUF model file
- `tts_model` — Coqui TTS model identifier (or use local `tts_model_path` + `tts_config_path`)
- `tts_enabled` — Enable/disable TTS playback
- `vad_threshold`, `min_silence_duration_ms`, `vad_pre_buffer_ms` — VAD sensitivity
- `llm_temperature`, `llm_n_ctx`, `llm_max_tokens` — LLM generation control
- `max_conversation_turns` — history trimming limit

## Conventions

- **Finnish-first** — UI labels, status messages, and the default system prompt are in Finnish
- **CUDA DLL setup** — must run before any CUDA-dependent imports (torch, llama_cpp, faster_whisper)
- **Deferred imports** — heavy libraries are imported inside worker threads to keep GUI startup instant
- **Thread safety** — GUI worker uses `QThread` with `QueuedConnection` signals
- **Self-trigger prevention** — after TTS playback, the audio queue is cleared and VAD is reset
- **Models directory** — `models/` (gitignored, created by `setup_models.py`)

## Related Repositories

- [voice-chatbot-ros](https://github.com/Aapo2001/voice-chatbot-ros) — ROS 2 Humble integration (depends on this pip package)
- [voice-chatbot-docs](https://github.com/Aapo2001/voice-chatbot-docs) — Documentation website ([live site](https://docs-site-kappa-coral.vercel.app))
