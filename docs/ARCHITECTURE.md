# Architecture

## Overview

The application is a local speech-to-speech assistant. Audio enters through the microphone, is segmented with voice activity detection, converted to text, answered by a local LLM, then synthesized back to speech.

Primary pipeline:

`AudioIO` -> `VoiceActivityDetector` -> `SpeechToText` -> `ChatLLM` -> `TextToSpeech` -> `AudioIO.play_audio`

Full documentation: [docs-site-kappa-coral.vercel.app](https://docs-site-kappa-coral.vercel.app)

## Entry Points

### `voice_chatbot/app.py`

Desktop UI built with PySide6.

- `MainWindow` builds the settings sidebar, chat view, log view, and toolbar controls.
- `ChatbotWorker` runs model initialization and the audio loop on a background `QThread`.
- `LogStream` redirects `stdout` and `stderr` into the GUI log panel.
- Text input bar allows sending messages directly to the LLM.
- TTS can be toggled on/off in settings without restarting.

This is the main operational interface for the project.

### `voice_chatbot/chatbot.py`

Terminal runner for the same pipeline.

- initializes all model wrappers in process
- prints status and chat messages to the console
- loops until interrupted with `Ctrl+C`
- loads persisted settings through `Config.load()`

## Module Responsibilities

### `voice_chatbot/config.py`

Defines the `Config` dataclass and JSON serialization helpers.

- stores runtime defaults for audio, VAD, STT, LLM, and TTS
- persists configuration with `save()`
- loads compatible fields from `config.json` with `load()`

The `load()` implementation ignores unknown JSON keys, which makes config evolution tolerant to stale fields.

### `voice_chatbot/audio_io.py`

Owns microphone capture and audio playback through `sounddevice`.

- opens an `InputStream`
- pushes chunks into a `queue.Queue`
- returns chunks with `get_audio_chunk()`
- blocks during playback with `sd.wait()`
- can clear pending captured chunks after TTS playback

The module uses the default input and output devices only.

### `voice_chatbot/vad.py`

Wraps Silero VAD and adds practical buffering logic.

Key behaviors:

- applies a simple RMS energy floor before speech starts
- maintains a pre-buffer so speech onset is not clipped
- concatenates captured chunks once Silero reports speech end
- drops utterances shorter than the configured minimum speech duration

This module is where turn segmentation quality is determined.

### `voice_chatbot/stt.py`

Loads Whisper through `faster_whisper.WhisperModel`.

- converts `int16` PCM to normalized `float32`
- transcribes with the configured language
- joins returned segments into a single string

### `voice_chatbot/llm.py`

Wraps `llama_cpp.Llama` for chat completion.

- loads a local GGUF model
- prepends a system prompt to each request
- stores alternating user and assistant messages
- trims history to `max_conversation_turns * 2` messages

### `voice_chatbot/tts_engine.py`

Loads Coqui TTS and synthesizes speech to a NumPy array.

- initializes `TTS(model_name=..., gpu=...)`
- exposes `synthesize(text)` returning `(audio, sample_rate)`
- includes a compatibility patch for transformers 5.x

### `voice_chatbot/ui_common.py`

Shared PySide6 UI components including the settings sidebar panel.

### `voice_chatbot/platform_setup.py`

CUDA DLL path setup and PySide6 DLL workarounds.

### `voice_chatbot/setup_models.py`

Pre-flight model setup script.

- validates PyTorch CUDA availability
- triggers model downloads and cache initialization
- downloads the configured GGUF file from Hugging Face if missing

This script loads `config.json` through `Config.load()`.

## GUI Runtime Sequence

When the user clicks `Käynnistä`:

1. `MainWindow` reads widget values into a fresh `Config`.
2. The config is written to `config.json`.
3. The LLM file path is validated.
4. `ChatbotWorker` starts in a background thread.
5. The worker initializes `AudioIO`, VAD, STT, LLM, and TTS.
6. The worker starts microphone capture and begins processing chunks.

During speech:

1. audio chunks enter the VAD
2. speech start changes GUI state to "Puhe havaittu..."
3. speech end triggers STT
4. transcribed text is shown in the chat panel
5. the LLM response is generated and shown
6. TTS audio is synthesized and played
7. microphone queue and VAD state are cleared before listening resumes

## State And Persistence

There are two distinct kinds of state:

- persisted configuration in `config.json`
- in-memory conversation history inside `ChatLLM`

The GUI "clear chat" action affects only the rendered chat panel. It does not clear the in-memory conversation history held by the active `ChatLLM` instance.

## Platform Assumptions

The code assumes Windows-oriented CUDA setup:

- `app.py` and `chatbot.py` add DLL directories before importing CUDA-dependent libraries
- the fallback path is `D:\cuda`
- the install scripts target CUDA 12.8 wheels for PyTorch
