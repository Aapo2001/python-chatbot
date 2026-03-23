"""Download and verify all required models before first run."""

import os
import sys
from pathlib import Path

from config import Config


def check_cuda():
    print("Checking CUDA availability...")
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  CUDA is available: {gpu_name}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("  WARNING: CUDA is NOT available. Models will run on CPU.")
            print("  Make sure you installed PyTorch with CUDA support.")
    except ImportError:
        print("  ERROR: PyTorch is not installed. Run install.bat first.")
        sys.exit(1)


def setup_whisper(config: Config):
    print(f"\nSetting up Whisper model '{config.whisper_model}'...")
    try:
        from pywhispercpp.model import Model

        model = Model(config.whisper_model)
        print(f"  Whisper model '{config.whisper_model}' is ready.")
        del model
    except Exception as e:
        print(f"  ERROR setting up Whisper: {e}")
        sys.exit(1)


def setup_llm(config: Config):
    print(f"\nSetting up LLM model...")
    model_path = Path(config.llm_model_path)

    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024**3)
        print(f"  LLM model already exists: {model_path} ({size_gb:.1f} GB)")
        return

    print(f"  Downloading '{config.llm_filename}' from '{config.llm_repo_id}'...")
    os.makedirs(config.models_dir, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(
            repo_id=config.llm_repo_id,
            filename=config.llm_filename,
            local_dir=config.models_dir,
        )
        print(f"  Downloaded to: {downloaded_path}")

        # Rename to expected path if different
        downloaded = Path(downloaded_path)
        if downloaded != model_path and downloaded.exists():
            downloaded.rename(model_path)
            print(f"  Renamed to: {model_path}")

    except Exception as e:
        print(f"  ERROR downloading LLM: {e}")
        print(f"  You can manually download from: https://huggingface.co/{config.llm_repo_id}")
        print(f"  Place the GGUF file at: {model_path}")
        sys.exit(1)


def setup_tts(config: Config):
    print(f"\nSetting up TTS model '{config.tts_model}'...")
    try:
        from TTS.api import TTS

        tts = TTS(model_name=config.tts_model, gpu=False)  # CPU for setup only
        print(f"  TTS model '{config.tts_model}' is ready.")
        del tts
    except Exception as e:
        print(f"  ERROR setting up TTS: {e}")
        sys.exit(1)


def setup_vad():
    print("\nSetting up Silero-VAD model...")
    try:
        from silero_vad import load_silero_vad

        model = load_silero_vad()
        print("  Silero-VAD model is ready.")
        del model
    except Exception as e:
        print(f"  ERROR setting up VAD: {e}")
        sys.exit(1)


def main():
    config = Config()

    print("=" * 50)
    print("  Voice Chatbot - Model Setup")
    print("=" * 50)

    check_cuda()
    setup_vad()
    setup_whisper(config)
    setup_llm(config)
    setup_tts(config)

    print("\n" + "=" * 50)
    print("  Setup complete!")
    print("=" * 50)
    print("\nAll models are downloaded and ready.")
    print("Run 'python chatbot.py' to start the voice chatbot.")


if __name__ == "__main__":
    main()
