"""
Model bootstrap script — download and validate all required models.

Run this **once** before the first launch to ensure every model file
is present and loadable.  The script validates each stage in order:

1. **CUDA** – check PyTorch can see the GPU.
2. **VAD** – download/cache Silero-VAD from ``torch.hub``.
3. **Whisper** – initialise the ``pywhispercpp`` model cache.
4. **LLM** – download the GGUF file from HuggingFace if missing.
5. **TTS** – initialise the Coqui TTS model cache.

Usage::

    pixi run setup-models
    # or: python setup_models.py
"""

import os
import sys
from pathlib import Path

from config import Config

# ── CUDA DLL setup (must run before torch / CUDA-backed imports) ──
_cuda_path = os.environ.get("CUDA_PATH", r"D:\cuda")
for _p in [os.path.join(_cuda_path, "bin", "x64"), os.path.join(_cuda_path, "bin")]:
    if hasattr(os, "add_dll_directory") and os.path.isdir(_p):
        os.add_dll_directory(_p)
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")


def check_cuda():
    """Verify that PyTorch is installed and can see a CUDA GPU."""
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
    """Load the configured Whisper model to warm up the cache."""
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
    """Download the GGUF model from HuggingFace if it is not already present."""
    print("\nSetting up LLM model...")
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

        # Rename to the expected path if hf_hub_download placed it elsewhere.
        downloaded = Path(downloaded_path)
        if downloaded != model_path and downloaded.exists():
            downloaded.rename(model_path)
            print(f"  Renamed to: {model_path}")

    except Exception as e:
        print(f"  ERROR downloading LLM: {e}")
        print(
            f"  You can manually download from: https://huggingface.co/{config.llm_repo_id}"
        )
        print(f"  Place the GGUF file at: {model_path}")
        sys.exit(1)


def setup_tts(config: Config):
    """Load the configured Coqui TTS model (CPU-only) to warm up the cache."""
    print(f"\nSetting up TTS model '{config.tts_model}'...")
    try:
        from TTS.api import TTS

        tts = TTS(model_name=config.tts_model, gpu=False)  # CPU-only for setup
        print(f"  TTS model '{config.tts_model}' is ready.")
        del tts
    except Exception as e:
        print(f"  ERROR setting up TTS: {e}")
        sys.exit(1)


def setup_vad():
    """Load Silero-VAD to ensure it is cached by ``torch.hub``."""
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
    """Run all model-setup stages sequentially."""
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
