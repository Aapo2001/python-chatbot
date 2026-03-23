from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_samples: int = 512  # 32ms at 16kHz

    # VAD settings
    vad_threshold: float = 0.5
    min_silence_duration_ms: int = 700
    speech_pad_ms: int = 30
    min_speech_duration_ms: int = 250

    # Language setting (used across STT, LLM, and TTS)
    language: str = "fi"  # Finnish

    # Whisper settings
    whisper_model: str = "base"  # multilingual model (no .en suffix)
    whisper_n_threads: int = 4

    # LLM settings
    models_dir: str = "models"
    llm_model_path: str = "models/mistral-7b-instruct-v0.3.Q4_K_M.gguf"
    llm_n_gpu_layers: int = -1  # -1 = offload all layers to GPU
    llm_n_ctx: int = 2048
    llm_max_tokens: int = 256
    llm_temperature: float = 0.7
    llm_system_prompt: str = (
        "Olet avulias suomenkielinen ääniassistentti. Vastaa aina suomeksi. "
        "Pidä vastauksesi lyhyinä ja keskustelunomaisina, sopivina puhuttavaksi "
        "ääneen. Rajoita vastauksesi 2-3 lauseeseen, ellei tarkempaa vastausta "
        "erikseen pyydetä."
    )
    max_conversation_turns: int = 20

    # TTS settings
    tts_model: str = "tts_models/fi/css10/vits"
    tts_gpu: bool = True

    # HuggingFace model repo for LLM download
    llm_repo_id: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    llm_filename: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
