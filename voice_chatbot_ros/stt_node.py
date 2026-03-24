"""
ROS 2 STT node – microphone capture, voice activity detection, and
speech-to-text transcription.

This is one of three split nodes in the ROS 2 architecture.  It owns
the microphone stream and publishes transcribed user text to the LLM
node.  The data flow is::

    Microphone → AudioIO → VAD → Whisper STT
                                      │
                    ┌─────────────────┘
                    ▼
          /voice_chatbot/user_text  ──→  LLM node
          /voice_chatbot/transcript ──→  GUI / logging

The node also subscribes to ``tts_done`` from the TTS node so it can
clear the microphone buffer and reset VAD after the assistant finishes
speaking (prevents self-triggering).
"""

import os
import threading
import traceback

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ── CUDA DLL setup ────────────────────────────────────────────────
_cuda_path = os.environ.get("CUDA_PATH", r"D:\cuda")
for _p in [os.path.join(_cuda_path, "bin", "x64"), os.path.join(_cuda_path, "bin")]:
    if hasattr(os, "add_dll_directory") and os.path.isdir(_p):
        os.add_dll_directory(_p)
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

from audio_io import AudioIO
from config import Config
from stt import SpeechToText
from vad import VoiceActivityDetector


class VoiceSttNode(Node):
    """ROS 2 node for real-time speech-to-text.

    Publishers:
        - ``user_text`` – transcribed text (consumed by the LLM node).
        - ``transcript`` – same text (consumed by the GUI for display).
        - ``status`` – pipeline state (``listening``, ``speech_detected``,
          ``transcribing``, ``error``).
        - ``log`` – human-readable log messages.

    Subscribers:
        - ``tts_done`` – signal from the TTS node that playback has
          finished, triggering a buffer clear + VAD reset.

    Parameters:
        - ``config_path`` (str) – path to ``config.json``.
        - ``load_config_file`` (bool) – whether to load the file or use
          defaults.
    """

    def __init__(self) -> None:
        super().__init__("voice_stt")

        self.declare_parameter("config_path", "config.json")
        self.declare_parameter("load_config_file", True)

        self._config = self._load_config()

        self._user_text_pub = self.create_publisher(String, "user_text", 10)
        self._transcript_pub = self.create_publisher(String, "transcript", 10)
        self._status_pub = self.create_publisher(String, "status", 10)
        self._log_pub = self.create_publisher(String, "log", 50)

        # Subscribe to TTS done signal to clear audio buffer and reset VAD
        self._tts_done_sub = self.create_subscription(
            String, "tts_done", self._on_tts_done, 10
        )

        self._running = threading.Event()
        self._running.set()
        self._voice_thread: threading.Thread | None = None

        self._audio: AudioIO | None = None
        self._vad: VoiceActivityDetector | None = None
        self._stt: SpeechToText | None = None

        self._initialize()

    def _load_config(self) -> Config:
        config_path = str(self.get_parameter("config_path").value)
        load_config_file = bool(self.get_parameter("load_config_file").value)
        if load_config_file:
            cfg = Config.load(config_path)
            self.get_logger().info(f"Loaded config from '{config_path}'.")
            return cfg
        self.get_logger().info("Using in-code Config defaults.")
        return Config()

    def _initialize(self) -> None:
        try:
            self._publish_status("initializing")
            self._publish_log("STT node: initializing audio capture...")
            self._audio = AudioIO(self._config)

            self._publish_log("STT node: initializing VAD...")
            self._vad = VoiceActivityDetector(self._config)

            self._publish_log("STT node: initializing Whisper STT...")
            self._stt = SpeechToText(self._config)

            self._audio.start_capture()
            self._voice_thread = threading.Thread(
                target=self._voice_loop, name="voice-capture-loop", daemon=True
            )
            self._voice_thread.start()

            self._publish_log("STT node ready.")
            self._publish_status("listening")
        except Exception as exc:
            self._publish_status("error")
            self._publish_log(f"STT node initialization failed: {exc}")
            self.get_logger().error(traceback.format_exc())
            raise

    def _on_tts_done(self, msg: String) -> None:
        """Clear audio buffer and reset VAD after TTS playback finishes."""
        del msg
        if self._audio is not None:
            self._audio.clear_queue()
        if self._vad is not None:
            self._vad.reset()
        self._publish_status("listening")

    def _voice_loop(self) -> None:
        assert self._audio is not None
        assert self._stt is not None
        assert self._vad is not None

        while self._running.is_set():
            try:
                chunk = self._audio.get_audio_chunk(timeout=0.1)
                if chunk is None:
                    continue

                event, audio_data = self._vad.process_chunk(chunk)
                if event == "speech_start":
                    self._publish_status("speech_detected")
                elif event == "speech_end":
                    self._publish_status("transcribing")
                    text = self._stt.transcribe(audio_data)
                    if not text or text.isspace():
                        self._publish_status("listening")
                        continue

                    self._transcript_pub.publish(String(data=text))
                    self._user_text_pub.publish(String(data=text))
                    self._publish_log(f"Transcript: {text}")
            except Exception:
                self._publish_status("error")
                self._publish_log("STT voice loop failed.")
                self.get_logger().error(traceback.format_exc())

    def _publish_status(self, status: str) -> None:
        self._status_pub.publish(String(data=status))
        self.get_logger().info(f"status={status}")

    def _publish_log(self, message: str) -> None:
        self._log_pub.publish(String(data=message))
        self.get_logger().info(message)

    def destroy_node(self) -> bool:
        self._running.clear()
        if self._audio is not None:
            self._audio.close()
        if self._voice_thread is not None and self._voice_thread.is_alive():
            self._voice_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    node: VoiceSttNode | None = None
    rclpy.init(args=args)
    try:
        node = VoiceSttNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
