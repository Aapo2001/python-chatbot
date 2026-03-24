"""
ROS 2 TTS node – text-to-speech synthesis and audio playback.

Subscribes to the assistant's reply text (from the LLM node),
synthesises speech with Coqui TTS, plays it through the speakers, and
publishes a ``tts_done`` signal so the STT node can reset its VAD.

Data flow::

    /voice_chatbot/assistant_text  ──→  Coqui TTS → Speaker
                                              │
                              ┌───────────────┘
                              ▼
                   /voice_chatbot/tts_done  ──→  STT node (VAD reset)

A synthesis queue + worker thread ensures that if the LLM publishes
multiple messages quickly, they are spoken in order without blocking
the ROS callback.
"""

import os
import queue
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
from tts_engine import TextToSpeech


class VoiceTtsNode(Node):
    """ROS 2 node for TTS synthesis and audio playback.

    Subscribers:
        - ``assistant_text`` – text to synthesise and speak.

    Publishers:
        - ``tts_done`` – emitted after playback finishes (triggers STT
          node VAD reset).
        - ``status`` – ``initializing``, ``ready``, ``speaking``,
          ``error``.
        - ``log`` – human-readable log messages.
    """

    def __init__(self) -> None:
        super().__init__("voice_tts")

        self.declare_parameter("config_path", "config.json")
        self.declare_parameter("load_config_file", True)

        self._config = self._load_config()

        self._assistant_sub = self.create_subscription(
            String, "assistant_text", self._on_assistant_text, 10
        )
        self._tts_done_pub = self.create_publisher(String, "tts_done", 10)
        self._status_pub = self.create_publisher(String, "status", 10)
        self._log_pub = self.create_publisher(String, "log", 50)

        self._synth_queue: queue.Queue[str] = queue.Queue()
        self._running = threading.Event()
        self._running.set()
        self._synth_thread: threading.Thread | None = None

        self._audio: AudioIO | None = None
        self._tts: TextToSpeech | None = None

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
            self._publish_log("TTS node: initializing audio playback...")
            self._audio = AudioIO(self._config)

            self._publish_log("TTS node: initializing TTS engine...")
            self._tts = TextToSpeech(self._config)

            self._synth_thread = threading.Thread(
                target=self._synth_loop, name="tts-synth-loop", daemon=True
            )
            self._synth_thread.start()

            self._publish_log("TTS node ready.")
            self._publish_status("ready")
        except Exception as exc:
            self._publish_status("error")
            self._publish_log(f"TTS node initialization failed: {exc}")
            self.get_logger().error(traceback.format_exc())
            raise

    def _on_assistant_text(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return
        self._synth_queue.put(text)

    def _synth_loop(self) -> None:
        while self._running.is_set():
            try:
                text = self._synth_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                assert self._tts is not None
                assert self._audio is not None

                self._publish_status("speaking")
                audio_out, sample_rate = self._tts.synthesize(text)
                self._audio.play_audio(audio_out, sample_rate)

                # Signal STT node that playback is done so it can reset VAD
                self._tts_done_pub.publish(String(data="done"))
                self._publish_status("ready")
            except Exception:
                self._publish_status("error")
                self._publish_log("TTS synthesis/playback failed.")
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
        if self._synth_thread is not None and self._synth_thread.is_alive():
            self._synth_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    node: VoiceTtsNode | None = None
    rclpy.init(args=args)
    try:
        node = VoiceTtsNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
