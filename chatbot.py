import os
import sys

# Add CUDA DLL directory to search path before any CUDA-dependent imports
_cuda_path = os.environ.get("CUDA_PATH", r"D:\cuda")
_cuda_bin_x64 = os.path.join(_cuda_path, "bin", "x64")
_cuda_bin = os.path.join(_cuda_path, "bin")
for _p in [_cuda_bin_x64, _cuda_bin]:
    if os.path.isdir(_p):
        os.add_dll_directory(_p)
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

from config import Config
from audio_io import AudioIO
from vad import VoiceActivityDetector
from stt import SpeechToText
from llm import ChatLLM
from tts_engine import TextToSpeech


class VoiceChatbot:
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

                    # Transcribe
                    text = self._stt.transcribe(audio_data)
                    if not text or text.isspace():
                        print("[Puhetta ei havaittu]")
                        continue

                    print(f"Sinä: {text}")

                    # Generate LLM response
                    response = self._llm.chat(text)
                    print(f"Botti: {response}")

                    # Synthesize and play speech
                    audio_out, sr = self._tts.synthesize(response)
                    self._audio.play_audio(audio_out, sr)

                    # Prevent hearing self: clear buffered audio and reset VAD
                    self._audio.clear_queue()
                    self._vad.reset()

        except KeyboardInterrupt:
            print("\n\nSammutetaan...")

        self._shutdown()

    def _shutdown(self):
        self._audio.close()
        print("Näkemiin!")


if __name__ == "__main__":
    chatbot = VoiceChatbot()
    chatbot.run()
