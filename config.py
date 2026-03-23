import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path

CONFIG_FILE = "config.json"


@dataclass
class Config:
    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_samples: int = 512  # 32ms at 16kHz

    # VAD settings
    vad_threshold: float = 0.45
    min_silence_duration_ms: int = 550
    speech_pad_ms: int = 200
    min_speech_duration_ms: int = 400
    vad_pre_buffer_ms: int = 500  # audio kept before VAD triggers

    # Language setting (used across STT, LLM, and TTS)
    language: str = "fi"  # Finnish

    # Whisper settings
    whisper_model: str = "small"  # multilingual model – 'small' much better for Finnish
    whisper_n_threads: int = 4

    # LLM settings
    models_dir: str = "models"
    llm_model_path: str = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    llm_n_gpu_layers: int = -1  # -1 = offload all layers to GPU
    llm_n_ctx: int = 2048
    llm_max_tokens: int = 256
    llm_temperature: float = 0.7
    llm_system_prompt: str = (
        "You are a Finnish-speaking voice assistant. You MUST ALWAYS respond "
        "in Finnish (suomi). NEVER respond in English or any other language. "
        "Keep your responses concise and conversational (2-3 sentences), "
        "suitable for spoken dialogue. If the user's input is unclear, ask "
        "them to repeat in Finnish. "
        "Olet suomenkielinen ääniassistentti. Vastaa aina suomeksi, lyhyesti "
        "ja selkeästi."
    )
    max_conversation_turns: int = 20

    # TTS settings
    tts_model: str = "tts_models/fi/css10/vits"
    tts_gpu: bool = True

    # HuggingFace model repo for LLM download
    llm_repo_id: str = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    llm_filename: str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

    def save(self, path: str = CONFIG_FILE) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str = CONFIG_FILE) -> "Config":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            valid_fields = {field.name for field in fields(cls)}
            filtered = {k: v for k, v in data.items() if k in valid_fields}
            return cls(**filtered)
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()
