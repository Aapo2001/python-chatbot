"""
Runtime configuration for the voice chatbot pipeline.

All settings are stored in a single ``Config`` dataclass that can be
serialised to / loaded from a JSON file (default: ``config.json``).
The GUI reads and writes this file; the CLI and ROS 2 nodes load it
at startup.

Sections
--------
- **Audio** – sample rate, channels, and chunk size for ``sounddevice``.
- **VAD** – Silero-VAD thresholds, silence/speech durations, pre-buffer.
- **Language** – ISO 639-1 code shared by STT, LLM, and TTS.
- **Whisper** – model size, threading, and GPU toggle.
- **LLM** – GGUF model path, GPU offloading, context window, generation
  parameters, system prompt, and conversation-turn limit.
- **TTS** – Coqui TTS model identifier or local model path, GPU toggle.
- **HuggingFace** – repo coordinates used by ``setup_models.py`` to
  download the GGUF model on first run.
"""

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

APP_DIR_NAME = "voice-chatbot"
CONFIG_ENV_VAR = "VOICE_CHATBOT_CONFIG"


def default_config_path() -> Path:
    """Return the per-user config file path.

    The old project-root ``config.json`` location made the current working
    directory part of the trust boundary. Store the file in the user's config
    directory instead, while allowing an explicit override for tests and power
    users.
    """
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return Path(override).expanduser()

    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / APP_DIR_NAME / "config.json"

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / APP_DIR_NAME / "config.json"

    return Path.home() / ".config" / APP_DIR_NAME / "config.json"


def resolve_config_path(path: str | Path | None = None) -> Path:
    """Resolve an explicit config path or the default per-user location."""
    return default_config_path() if path is None else Path(path).expanduser()


@dataclass
class Config:
    """Central configuration dataclass for every pipeline component.

    Default values match a Finnish-language setup with GPU acceleration.
    The GUI's ``SettingsPanel`` maps every widget to one of these fields
    and calls :meth:`save` when the user presses *Start* or *Save*.

    The :meth:`load` classmethod silently ignores unknown keys in the
    JSON file so that old config files remain forward-compatible when
    new fields are added.
    """

    # ── Audio settings ────────────────────────────────────────────
    sample_rate: int = 16000  # Hz – matches Whisper's expected input
    channels: int = 1  # mono microphone capture
    chunk_samples: int = 512  # ≈ 32 ms at 16 kHz (Silero VAD window)

    # ── VAD (Voice Activity Detection) settings ───────────────────
    vad_threshold: float = 0.45  # Silero confidence threshold (0–1)
    min_silence_duration_ms: int = 550  # silence required to end an utterance
    speech_pad_ms: int = 200  # extra audio kept around speech edges
    min_speech_duration_ms: int = 400  # ignore utterances shorter than this
    vad_pre_buffer_ms: int = 500  # audio kept before VAD triggers
    # The pre-buffer prevents the first syllable from being clipped
    # when Silero detects speech slightly after it has already begun.

    # ── Language setting (used across STT, LLM, and TTS) ─────────
    language: str = "fi"  # ISO 639-1 code; "fi" = Finnish

    # ── Whisper STT settings ──────────────────────────────────────
    whisper_model: str = "medium"  # tiny / base / small / medium / large-v*
    whisper_n_threads: int = 4  # CPU threads for CTranslate2
    whisper_gpu: bool = True  # use CUDA if available

    # ── LLM (Large Language Model) settings ───────────────────────
    models_dir: str = "models"  # directory for downloaded models
    llm_model_path: str = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    llm_n_gpu_layers: int = -1  # -1 = offload ALL layers to GPU
    llm_n_ctx: int = 2048  # context window size (tokens)
    llm_max_tokens: int = 256  # max tokens per assistant reply
    llm_temperature: float = 0.7  # sampling temperature (0 = greedy)
    llm_system_prompt: str = (
        # Bilingual system prompt: English instructions followed by a
        # Finnish summary, to reinforce Finnish-only responses.
        "You are a Finnish-speaking voice assistant. You MUST ALWAYS respond "
        "in Finnish (suomi). NEVER respond in English or any other language. "
        "Keep your responses concise and conversational (2-3 sentences), "
        "suitable for spoken dialogue. If the user's input is unclear, ask "
        "them to repeat in Finnish. "
        "Olet suomenkielinen ääniassistentti. Vastaa aina suomeksi, lyhyesti "
        "ja selkeästi."
    )
    max_conversation_turns: int = 20  # oldest turn pairs are trimmed beyond this

    # ── TTS (Text-to-Speech) settings ─────────────────────────────
    tts_model: str = "tts_models/fi/css10/vits"  # Coqui model identifier
    tts_model_path: str = "models/model.pth"  # local VITS checkpoint
    tts_config_path: str = "models/config.json"  # local VITS config
    tts_gpu: bool = True  # use CUDA for TTS synthesis
    tts_enabled: bool = True  # enable TTS playback (disable for text-only mode)

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str | Path | None = None) -> None:
        """Serialise the current config to a JSON file.

        Args:
            path: Output file path. Defaults to the per-user config file.
        """
        target = resolve_config_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        tmp_path.replace(target)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load a config from a JSON file, falling back to defaults.

        Keys in the JSON that do not correspond to a dataclass field
        are silently discarded so that older config files still work
        after new fields are added.

        Args:
            path: Input file path. Defaults to the per-user config file.

        Returns:
            A populated :class:`Config` instance.
        """
        source = resolve_config_path(path)
        try:
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Filter out any keys that are not declared fields
            valid_fields = {field.name for field in fields(cls)}
            filtered = {k: v for k, v in data.items() if k in valid_fields}
            return cls(**filtered)
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()
