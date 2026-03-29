"""
Voice Chatbot — local speech-to-speech voice assistant.

Install with extras for full functionality::

    pip install voice-chatbot[all]

Or pick components individually::

    pip install voice-chatbot[stt,llm,tts,vad,gui]
"""

__version__ = "0.1.1"

from .config import Config

__all__ = ["Config", "__version__"]
