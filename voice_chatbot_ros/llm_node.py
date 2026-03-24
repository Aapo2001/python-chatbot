"""
ROS 2 LLM node – multi-turn chat inference.

Subscribes to transcribed user text (from the STT node or the GUI) and
publishes the assistant's reply for the TTS node to synthesise.

Data flow::

    /voice_chatbot/user_text  ──→  ChatLLM.chat()
                                       │
                       ┌───────────────┘
                       ▼
    /voice_chatbot/assistant_text  ──→  TTS node

The node is stateless except for the in-memory conversation history,
which can be cleared via the ``clear_history`` service.  A request
queue + worker thread serialises access to the :class:`ChatLLM`
instance so that concurrent messages don't race.
"""

import os
import queue
import threading
import traceback

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

# ── CUDA DLL setup ────────────────────────────────────────────────
_cuda_path = os.environ.get("CUDA_PATH", r"D:\cuda")
for _p in [os.path.join(_cuda_path, "bin", "x64"), os.path.join(_cuda_path, "bin")]:
    if hasattr(os, "add_dll_directory") and os.path.isdir(_p):
        os.add_dll_directory(_p)
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

from config import Config
from llm import ChatLLM


class VoiceLlmNode(Node):
    """ROS 2 node for LLM chat inference.

    Subscribers:
        - ``user_text`` – incoming user messages.

    Publishers:
        - ``assistant_text`` – LLM replies (consumed by the TTS node).
        - ``status`` – ``initializing``, ``ready``, ``llm_responding``,
          ``error``.
        - ``log`` – human-readable log messages.

    Services:
        - ``clear_history`` (``Trigger``) – erase conversation memory.
    """

    def __init__(self) -> None:
        super().__init__("voice_llm")

        self.declare_parameter("config_path", "config.json")
        self.declare_parameter("load_config_file", True)

        self._config = self._load_config()

        self._user_sub = self.create_subscription(
            String, "user_text", self._on_user_text, 10
        )
        self._assistant_pub = self.create_publisher(String, "assistant_text", 10)
        self._status_pub = self.create_publisher(String, "status", 10)
        self._log_pub = self.create_publisher(String, "log", 50)
        self._clear_srv = self.create_service(
            Trigger, "clear_history", self._on_clear_history
        )

        self._request_queue: queue.Queue[str] = queue.Queue()
        self._running = threading.Event()
        self._running.set()
        self._request_thread: threading.Thread | None = None

        self._llm: ChatLLM | None = None

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
            self._publish_log("LLM node: initializing LLM...")
            self._llm = ChatLLM(self._config)

            self._request_thread = threading.Thread(
                target=self._request_loop, name="chat-request-loop", daemon=True
            )
            self._request_thread.start()

            self._publish_log("LLM node ready.")
            self._publish_status("ready")
        except Exception as exc:
            self._publish_status("error")
            self._publish_log(f"LLM node initialization failed: {exc}")
            self.get_logger().error(traceback.format_exc())
            raise

    def _on_user_text(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            self._publish_log("Ignoring empty user_text message.")
            return
        self._publish_log(f"Queued user text: {text}")
        self._request_queue.put(text)

    def _on_clear_history(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        if self._llm is None:
            response.success = False
            response.message = "LLM is not initialized."
            return response
        self._llm.clear_history()
        self._publish_log("Conversation history cleared.")
        response.success = True
        response.message = "Conversation history cleared."
        return response

    def _request_loop(self) -> None:
        while self._running.is_set():
            try:
                user_text = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if self._llm is None:
                    self._publish_log("Skipping request because LLM is unavailable.")
                    continue

                self._publish_status("llm_responding")
                response = self._llm.chat(user_text)
                self._assistant_pub.publish(String(data=response))
                self._publish_log(f"Assistant: {response}")
                self._publish_status("ready")
            except Exception:
                self._publish_status("error")
                self._publish_log("LLM request processing failed.")
                self.get_logger().error(traceback.format_exc())

    def _publish_status(self, status: str) -> None:
        self._status_pub.publish(String(data=status))
        self.get_logger().info(f"status={status}")

    def _publish_log(self, message: str) -> None:
        self._log_pub.publish(String(data=message))
        self.get_logger().info(message)

    def destroy_node(self) -> bool:
        self._running.clear()
        if self._request_thread is not None and self._request_thread.is_alive():
            self._request_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    node: VoiceLlmNode | None = None
    rclpy.init(args=args)
    try:
        node = VoiceLlmNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
