# Voice Chatbot

Local speech-to-speech voice assistant with PySide6 GUI and CLI. Captures microphone audio, detects speech with Silero VAD, transcribes with Whisper, generates a reply with a local GGUF LLM, and speaks the reply with Coqui TTS.

Available on PyPI: `pip install voice-chatbot[all]`

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

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended for real-time inference)
- espeak-ng (required for Coqui TTS phonemisation)
  - Windows: install to `C:\Program Files\eSpeak NG`
  - Linux: `sudo apt install espeak-ng`
- CUDA toolkit — DLL path defaults to `D:\cuda` (override with `CUDA_PATH` env var)

### pip install

```bash
# Full installation (all components)
pip install voice-chatbot[all]

# Or pick specific components
pip install voice-chatbot[stt,llm,tts,vad,gui]

# Core only (numpy, sounddevice, huggingface-hub)
pip install voice-chatbot
```

### Pixi install (from source)

```bash
pixi install
pixi run install-python-deps
```

### Download models

```bash
# Via pip entry point
voice-chatbot-setup-models

# Or via pixi
pixi run setup-models
```

Models are downloaded to `models/` (gitignored).

## Usage

### Desktop GUI (primary)

```bash
voice-chatbot-app
# or: pixi run app
# or: python -m voice_chatbot.app
```

The GUI provides a settings sidebar, chat panel with text input, system log, and start/stop/restart controls. Text input lets you type messages directly to the LLM. TTS can be toggled on/off in settings without restarting.

### CLI (headless)

```bash
voice-chatbot
# or: pixi run chatbot
# or: python -m voice_chatbot.chatbot
```

## Project Structure

```
voice_chatbot/          Main Python package
├── app.py              PySide6 desktop GUI
├── chatbot.py          Headless CLI runner
├── config.py           Config dataclass + JSON persistence
├── audio_io.py         Microphone capture + playback (sounddevice)
├── vad.py              Silero-VAD wrapper with pre-buffer
├── stt.py              Whisper STT (faster-whisper / CTranslate2)
├── llm.py              LLaMA multi-turn chat (llama-cpp-python)
├── tts_engine.py       Coqui TTS wrapper
├── ui_common.py        Shared PySide6 UI components
├── platform_setup.py   CUDA / PySide6 DLL setup
└── setup_models.py     Model download + validation

tests/                  Pytest test suite (36 tests)
tools/
├── install_python_windows.bat
└── install_python_linux.sh

pyproject.toml          Package metadata + build config
pixi.toml               Pixi workspace manifest
config.json             Runtime configuration (saved by GUI)
```

## Configuration

All runtime settings are in `config.json`. The GUI reads/writes this file. Key fields:

- `language` — STT/TTS language code (default: `"fi"`)
- `whisper_model` — Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`)
- `llm_model_path` — Path to GGUF model file
- `tts_model` — Coqui TTS model identifier
- `tts_enabled` — Enable/disable TTS playback
- `vad_threshold`, `min_silence_duration_ms`, `vad_pre_buffer_ms` — VAD sensitivity
- `llm_temperature`, `llm_n_ctx`, `llm_max_tokens` — LLM generation control
- `max_conversation_turns` — History trimming limit

## Testing

```bash
pixi run test
# or: pytest
```

## Related Repositories

- [voice-chatbot-ros](https://github.com/Aapo2001/voice-chatbot-ros) — ROS 2 Humble integration (depends on this pip package)
- [voice-chatbot-docs](https://github.com/Aapo2001/voice-chatbot-docs) — Documentation website ([live site](https://docs-site-kappa-coral.vercel.app))

## License

MIT
