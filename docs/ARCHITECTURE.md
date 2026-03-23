# Architecture

## Overview

The application is a local speech-to-speech assistant. Audio enters through the microphone, is segmented with voice activity detection, converted to text, answered by a local LLM, then synthesized back to speech.

Primary pipeline:

`AudioIO` -> `VoiceActivityDetector` -> `SpeechToText` -> `ChatLLM` -> `TextToSpeech` -> `AudioIO.play_audio`

## Entry Points

### `app.py`

Desktop UI built with PySide6.

- `MainWindow` builds the settings sidebar, chat view, log view, and toolbar controls.
- `ChatbotWorker` runs model initialization and the audio loop on a background `QThread`.
- `LogStream` redirects `stdout` and `stderr` into the GUI log panel.

This is the main operational interface for the project.

### `chatbot.py`

Terminal runner for the same pipeline.

- initializes all model wrappers in process
- prints status and chat messages to the console
- loops until interrupted with `Ctrl+C`

Unlike the GUI path, it currently uses `Config()` defaults rather than loading `config.json`.

### `voice_chatbot_ros/node.py`

ROS 2 Humble integration node.

- exposes text and status topics over `rclpy`
- optionally runs the microphone, VAD, and STT loop in a background thread
- reuses the same `ChatLLM` and `TextToSpeech` wrappers as the desktop and CLI entry points
- offers a `clear_history` service for resetting the in-memory conversation state

## Module Responsibilities

### `config.py`

Defines the `Config` dataclass and JSON serialization helpers.

- stores runtime defaults for audio, VAD, STT, LLM, and TTS
- persists configuration with `save()`
- loads compatible fields from `config.json` with `load()`

The `load()` implementation ignores unknown JSON keys, which makes config evolution tolerant to stale fields.

### `audio_io.py`

Owns microphone capture and audio playback through `sounddevice`.

- opens an `InputStream`
- pushes chunks into a `queue.Queue`
- returns chunks with `get_audio_chunk()`
- blocks during playback with `sd.wait()`
- can clear pending captured chunks after TTS playback

The module uses the default input and output devices only.

### `vad.py`

Wraps Silero VAD and adds practical buffering logic.

Key behaviors:

- applies a simple RMS energy floor before speech starts
- maintains a pre-buffer so speech onset is not clipped
- concatenates captured chunks once Silero reports speech end
- drops utterances shorter than the configured minimum speech duration

This module is where turn segmentation quality is determined.

### `stt.py`

Loads Whisper through `pywhispercpp.model.Model`.

- converts `int16` PCM to normalized `float32`
- transcribes with the configured language
- joins returned segments into a single string

### `llm.py`

Wraps `llama_cpp.Llama` for chat completion.

- loads a local GGUF model
- prepends a system prompt to each request
- stores alternating user and assistant messages
- trims history to `max_conversation_turns * 2` messages

`clear_history()` exists but is not called from the GUI controls.

### `tts_engine.py`

Loads Coqui TTS and synthesizes speech to a NumPy array.

- initializes `TTS(model_name=..., gpu=...)`
- exposes `synthesize(text)` returning `(audio, sample_rate)`

### `setup_models.py`

Pre-flight model setup script.

- validates PyTorch CUDA availability
- triggers model downloads and cache initialization
- downloads the configured GGUF file from Hugging Face if missing

This script uses config defaults rather than reading `config.json`.

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

The ROS node can clear the same in-memory history through `~/clear_history`.

## Platform Assumptions

The code assumes Windows-oriented CUDA setup:

- `app.py` and `chatbot.py` add DLL directories before importing CUDA-dependent libraries
- the fallback path is `D:\cuda`
- `install.bat` targets CUDA 12.8 wheels for PyTorch and CUDA-enabled local builds for native packages

## Notable Gaps

- no automated tests
- no device selection UI or config for audio input/output
- no streaming token generation or streaming TTS
- no explicit cancellation once a single LLM or TTS request is in progress
