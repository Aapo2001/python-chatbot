"""
Voice Chatbot – headless command-line interface.

Runs the complete speech-to-speech pipeline in a single process without
a GUI.  Useful for headless or SSH-based deployments, quick testing, or
when the PySide6 GUI is not needed.

Pipeline (sequential, single-threaded)::

    Microphone → VAD → Whisper STT → LLM → Coqui TTS → Speaker

Press **Ctrl+C** to quit gracefully.

Usage::

    python chatbot.py
"""

import os

# ── CUDA DLL setup (must run before any CUDA-dependent imports) ───
_cuda_path = os.environ.get("CUDA_PATH", r"D:\cuda")
_cuda_bin_x64 = os.path.join(_cuda_path, "bin", "x64")
_cuda_bin = os.path.join(_cuda_path, "bin")
for _p in [_cuda_bin_x64, _cuda_bin]:
    if os.path.isdir(_p):
        os.add_dll_directory(_p)
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

from audio_io import AudioIO
from config import Config
from llm import ChatLLM
from stt import SpeechToText
from tts_engine import TextToSpeech
from vad import VoiceActivityDetector


class VoiceChatbot:
    """Synchronous voice chatbot that loads all models and runs an audio loop.

    All five pipeline stages (audio, VAD, STT, LLM, TTS) are initialised
    in the constructor.  Call :meth:`run` to start the blocking audio loop.
    """

    def __init__(self):
        self._config = Config()

        print("=" * 50)
        print("  Äänichatbot - Paikallinen GPU-kiihdytetty")
        print("=" * 50)
        print()

        print("[Init] Setting up audio I/O...")
        self._audio = AudioIO(self._config)

        print("[Init] Loading VAD model...")
        self._vad = VoiceActivityDetector(self._config)
        print("[Init] VAD ready.")

        self._stt = SpeechToText(self._config)
        self._llm = ChatLLM(self._config)
        self._tts = TextToSpeech(self._config)

        print()
        print("Kaikki mallit ladattu onnistuneesti!")
        print()

    def run(self):
        """Start the blocking audio capture → process → speak loop."""
        self._audio.start_capture()
        print("Kuunnellaan... (paina Ctrl+C lopettaaksesi)")
        print("-" * 50)

        try:
            while True:
                chunk = self._audio.get_audio_chunk(timeout=0.1)
                if chunk is None:
                    continue

                event, audio_data = self._vad.process_chunk(chunk)

                if event == "speech_start":
                    print("\n[Kuunnellaan...]")

                elif event == "speech_end":
                    print("[Käsitellään...]")
                    if audio_data is not None:
                        # 1. Transcribe speech to text
                        text = self._stt.transcribe(audio_data)
                        if not text or text.isspace():
                            print("[Puhetta ei havaittu]")
                            continue

                        print(f"Sinä: {text}")

                        # 2. Generate LLM response
                        response = self._llm.chat(text)
                        print(f"Botti: {response}")

                        # 3. Synthesize and play speech
                        audio_out, sr = self._tts.synthesize(response)
                        self._audio.play_audio(audio_out, sr)

                        # 4. Prevent self-triggering: discard mic audio
                        #    recorded during playback and reset VAD state.
                        self._audio.clear_queue()
                        self._vad.reset()

        except KeyboardInterrupt:
            print("\n\nSammutetaan...")

        self._shutdown()

    def _shutdown(self):
        """Release audio resources."""
        self._audio.close()
        print("Näkemiin!")


if __name__ == "__main__":
    chatbot = VoiceChatbot()
    chatbot.run()
