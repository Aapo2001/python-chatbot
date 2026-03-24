"""
Voice Chatbot – ROS 2 PySide6 desktop GUI.

Unlike the standalone ``app.py``, this GUI does **not** load any ML
models itself.  Instead it connects to the three split ROS 2 nodes
(STT, LLM, TTS) running in separate processes and communicates via
ROS 2 topics and services.

ROS 2 topic/service interface (all in ``/voice_chatbot/`` namespace):

====================  ============  =================================
Topic / Service       Direction     Purpose
====================  ============  =================================
``log``               subscribe     System log messages from all nodes
``status``            subscribe     Pipeline state (``listening``, …)
``transcript``        subscribe     User speech transcription
``assistant_text``    subscribe     LLM reply text
``user_text``         **publish**   Send typed text to the LLM node
``clear_history``     service call  Clear LLM conversation memory
====================  ============  =================================

A background :class:`QThread` spins the ROS 2 node so that callbacks
fire without blocking the Qt event loop.

Usage::

    pixi run ros-app
    # or: python ros_app.py
"""

import os
import sys
from pathlib import Path

# ── PySide6 DLL workaround (must run before Qt widget imports) ─────
_site_packages = [Path(p) for p in sys.path if "site-packages" in p]
_pyside_dir = None
for _pkg_name in ("PySide6", "shiboken6"):
    for _base in _site_packages:
        _dll_dir = _base / _pkg_name
        if hasattr(os, "add_dll_directory") and _dll_dir.is_dir():
            os.add_dll_directory(str(_dll_dir))
            os.environ["PATH"] = str(_dll_dir) + os.pathsep + os.environ.get("PATH", "")
            if _pkg_name == "PySide6":
                _pyside_dir = _dll_dir

if _pyside_dir is not None:
    _plugins_dir = _pyside_dir / "plugins"
    _platforms_dir = _plugins_dir / "platforms"
    os.environ["QT_PLUGIN_PATH"] = str(_plugins_dir)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_platforms_dir)

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import rclpy
from rclpy.node import Node as RosNode
from std_msgs.msg import String
from std_srvs.srv import Trigger

from config import Config

# ── Constants ──────────────────────────────────────────────────────

WHISPER_MODELS = [
    "tiny", "tiny.en", "base", "base.en",
    "small", "small.en", "medium", "medium.en",
    "large-v1", "large-v2", "large-v3",
]

LANGUAGES = {
    "Suomi (fi)": "fi", "English (en)": "en", "Auto": "auto",
    "Svenska (sv)": "sv", "Deutsch (de)": "de", "Francais (fr)": "fr",
    "Espanol (es)": "es", "Italiano (it)": "it", "Portugues (pt)": "pt",
    "Nederlands (nl)": "nl", "Polski (pl)": "pl", "Japanese (ja)": "ja",
    "Chinese (zh)": "zh", "Korean (ko)": "ko", "Russian (ru)": "ru",
}

TTS_MODELS = [
    "tts_models/fi/css10/vits",
    "tts_models/en/ljspeech/tacotron2-DDC",
    "tts_models/en/ljspeech/vits",
    "tts_models/en/vctk/vits",
    "tts_models/de/thorsten/tacotron2-DDC",
    "tts_models/de/thorsten/vits",
    "tts_models/fr/mai/tacotron2-DDC",
    "tts_models/es/mai/tacotron2-DDC",
    "tts_models/nl/mai/tacotron2-DDC",
    "tts_models/ja/kokoro/tacotron2-DDC",
    "tts_models/zh-CN/baker/tacotron2-DDC-GST",
    "tts_models/multilingual/multi-dataset/xtts_v2",
]

# Map ROS status strings → Finnish UI text
_STATUS_MAP = {
    "initializing": "Alustetaan...",
    "ready": "Valmis",
    "listening": "Kuunnellaan...",
    "speech_detected": "Puhe havaittu...",
    "transcribing": "Käsitellään...",
    "llm_responding": "LLM vastaa...",
    "speaking": "Puhutaan...",
    "error": "Virhe",
}


# ── ROS 2 spin thread ─────────────────────────────────────────────


class _RosSpinThread(QThread):
    """Spins the ROS 2 node in a background thread.

    Calls ``rclpy.spin_once`` in a tight loop (50 ms timeout) so that
    subscription callbacks and service responses are processed without
    blocking the Qt event loop.
    """

    def __init__(self, node: RosNode, parent: QObject | None = None):
        super().__init__(parent)
        self._node = node
        self._running = True

    def run(self) -> None:
        while self._running and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.05)

    def stop(self) -> None:
        self._running = False


# ── ROS 2 ↔ Qt bridge ─────────────────────────────────────────────


class RosBridge(QObject):
    """Bridges ROS 2 topics/services to Qt signals.

    Creates a lightweight ``rclpy`` node (``voice_chatbot_gui``) that
    subscribes to the split-node topics and emits Qt signals when
    messages arrive.  Also provides :meth:`send_text` (publish) and
    :meth:`clear_history` (service call) for outbound commands.

    The ROS spin loop runs in a :class:`_RosSpinThread`.  ROS
    callbacks emit Qt signals which, via ``QueuedConnection``, are
    delivered on the main thread — safe for direct UI updates.
    """

    log_received = Signal(str)
    status_received = Signal(str)
    chat_message = Signal(str, str)  # (role, text)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._node: RosNode | None = None
        self._spin_thread: _RosSpinThread | None = None
        self._user_pub = None
        self._clear_client = None

    @property
    def is_connected(self) -> bool:
        return self._node is not None

    def connect_ros(self) -> None:
        if self._node is not None:
            return

        rclpy.init()
        self._node = RosNode("voice_chatbot_gui")

        # Subscriptions (absolute topic names matching the split nodes' namespace)
        self._node.create_subscription(
            String, "/voice_chatbot/log", self._on_log, 50
        )
        self._node.create_subscription(
            String, "/voice_chatbot/status", self._on_status, 10
        )
        self._node.create_subscription(
            String, "/voice_chatbot/transcript", self._on_transcript, 10
        )
        self._node.create_subscription(
            String, "/voice_chatbot/assistant_text", self._on_assistant, 10
        )

        # Publisher for sending typed text
        self._user_pub = self._node.create_publisher(
            String, "/voice_chatbot/user_text", 10
        )

        # Service client for clearing LLM history
        self._clear_client = self._node.create_client(
            Trigger, "/voice_chatbot/clear_history"
        )

        self._spin_thread = _RosSpinThread(self._node)
        self._spin_thread.start()

    def disconnect_ros(self) -> None:
        if self._node is None:
            return

        if self._spin_thread is not None:
            self._spin_thread.stop()
            self._spin_thread.wait(3000)
            self._spin_thread = None

        self._user_pub = None
        self._clear_client = None
        self._node.destroy_node()
        self._node = None
        rclpy.try_shutdown()

    def send_text(self, text: str) -> None:
        if self._user_pub is not None:
            self._user_pub.publish(String(data=text))
            self.chat_message.emit("user", text)

    def clear_history(self) -> None:
        if self._clear_client is not None and self._clear_client.service_is_ready():
            self._clear_client.call_async(Trigger.Request())
            self.log_received.emit("Keskusteluhistoria tyhjennetty.")

    # ── ROS callbacks (called from spin thread) ──

    def _on_log(self, msg: String) -> None:
        self.log_received.emit(msg.data)

    def _on_status(self, msg: String) -> None:
        text = _STATUS_MAP.get(msg.data, msg.data)
        self.status_received.emit(text)

    def _on_transcript(self, msg: String) -> None:
        self.chat_message.emit("user", msg.data)

    def _on_assistant(self, msg: String) -> None:
        self.chat_message.emit("assistant", msg.data)


# ── Settings panel ─────────────────────────────────────────────────


class SettingsPanel(QScrollArea):
    """Left sidebar with all editable settings.

    Identical widget set to the standalone ``app.py`` panel.  Changes
    are only written to ``config.json`` when the user clicks *Save* —
    they take effect after the ROS 2 nodes are restarted.
    """

    def __init__(self, config: Config, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(310)
        self.setMaximumWidth(420)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── Language & models ──
        grp_models = QGroupBox("Kieli ja mallit")
        form_models = QFormLayout(grp_models)

        self.combo_language = QComboBox()
        for label, code in LANGUAGES.items():
            self.combo_language.addItem(label, code)
        form_models.addRow("Kieli:", self.combo_language)

        self.combo_whisper = QComboBox()
        self.combo_whisper.addItems(WHISPER_MODELS)
        form_models.addRow("Whisper-malli:", self.combo_whisper)

        llm_row = QHBoxLayout()
        self.edit_llm_path = QLineEdit()
        self.edit_llm_path.setPlaceholderText("polku .gguf-tiedostoon")
        self.btn_browse_llm = QPushButton("...")
        self.btn_browse_llm.setFixedWidth(32)
        self.btn_browse_llm.setToolTip("Selaa GGUF-tiedostoja")
        self.btn_browse_llm.clicked.connect(self._browse_llm)
        llm_row.addWidget(self.edit_llm_path)
        llm_row.addWidget(self.btn_browse_llm)
        form_models.addRow("LLM-malli:", llm_row)

        self.combo_tts = QComboBox()
        self.combo_tts.setEditable(True)
        self.combo_tts.addItems(TTS_MODELS)
        form_models.addRow("TTS-malli:", self.combo_tts)

        layout.addWidget(grp_models)

        # ── LLM settings ──
        grp_llm = QGroupBox("LLM-asetukset")
        form_llm = QFormLayout(grp_llm)

        self.spin_temperature = QDoubleSpinBox()
        self.spin_temperature.setRange(0.0, 2.0)
        self.spin_temperature.setSingleStep(0.1)
        self.spin_temperature.setDecimals(2)
        form_llm.addRow("Lämpötila:", self.spin_temperature)

        self.spin_max_tokens = QSpinBox()
        self.spin_max_tokens.setRange(32, 4096)
        self.spin_max_tokens.setSingleStep(32)
        form_llm.addRow("Max tokenit:", self.spin_max_tokens)

        self.spin_ctx = QSpinBox()
        self.spin_ctx.setRange(512, 131072)
        self.spin_ctx.setSingleStep(512)
        form_llm.addRow("Konteksti (n_ctx):", self.spin_ctx)

        self.spin_gpu_layers = QSpinBox()
        self.spin_gpu_layers.setRange(-1, 200)
        self.spin_gpu_layers.setSpecialValueText("Kaikki (-1)")
        form_llm.addRow("GPU-tasot:", self.spin_gpu_layers)

        self.spin_turns = QSpinBox()
        self.spin_turns.setRange(1, 100)
        form_llm.addRow("Max keskusteluvuorot:", self.spin_turns)

        self.edit_system_prompt = QPlainTextEdit()
        self.edit_system_prompt.setFixedHeight(100)
        self.edit_system_prompt.setPlaceholderText("Järjestelmäkehote...")
        form_llm.addRow("Järjestelmäkehote:", self.edit_system_prompt)

        layout.addWidget(grp_llm)

        # ── VAD settings ──
        grp_vad = QGroupBox("VAD-asetukset")
        form_vad = QFormLayout(grp_vad)

        self.spin_vad_threshold = QDoubleSpinBox()
        self.spin_vad_threshold.setRange(0.05, 1.0)
        self.spin_vad_threshold.setSingleStep(0.05)
        self.spin_vad_threshold.setDecimals(2)
        form_vad.addRow("Kynnysarvo:", self.spin_vad_threshold)

        self.spin_silence_ms = QSpinBox()
        self.spin_silence_ms.setRange(100, 5000)
        self.spin_silence_ms.setSingleStep(50)
        self.spin_silence_ms.setSuffix(" ms")
        form_vad.addRow("Hiljaisuus (min):", self.spin_silence_ms)

        self.spin_speech_pad = QSpinBox()
        self.spin_speech_pad.setRange(0, 500)
        self.spin_speech_pad.setSingleStep(10)
        self.spin_speech_pad.setSuffix(" ms")
        form_vad.addRow("Puhetäyte:", self.spin_speech_pad)

        self.spin_min_speech = QSpinBox()
        self.spin_min_speech.setRange(50, 5000)
        self.spin_min_speech.setSingleStep(50)
        self.spin_min_speech.setSuffix(" ms")
        form_vad.addRow("Puhe (min):", self.spin_min_speech)

        self.spin_pre_buffer = QSpinBox()
        self.spin_pre_buffer.setRange(100, 2000)
        self.spin_pre_buffer.setSingleStep(50)
        self.spin_pre_buffer.setSuffix(" ms")
        form_vad.addRow("Esipuskuri:", self.spin_pre_buffer)

        layout.addWidget(grp_vad)

        # ── TTS extra ──
        grp_tts = QGroupBox("TTS-asetukset")
        form_tts = QFormLayout(grp_tts)
        self.chk_tts_gpu = QCheckBox("Käytä GPU:ta")
        form_tts.addRow(self.chk_tts_gpu)
        layout.addWidget(grp_tts)

        layout.addStretch()
        self.setWidget(container)

        self.load_from_config(config)

    def load_from_config(self, cfg: Config) -> None:
        _set_combo_by_data(self.combo_language, cfg.language)
        _set_combo_by_text(self.combo_whisper, cfg.whisper_model)
        self.edit_llm_path.setText(cfg.llm_model_path)
        _set_combo_by_text(self.combo_tts, cfg.tts_model)
        self.spin_temperature.setValue(cfg.llm_temperature)
        self.spin_max_tokens.setValue(cfg.llm_max_tokens)
        self.spin_ctx.setValue(cfg.llm_n_ctx)
        self.spin_gpu_layers.setValue(cfg.llm_n_gpu_layers)
        self.spin_turns.setValue(cfg.max_conversation_turns)
        self.edit_system_prompt.setPlainText(cfg.llm_system_prompt)
        self.spin_vad_threshold.setValue(cfg.vad_threshold)
        self.spin_silence_ms.setValue(cfg.min_silence_duration_ms)
        self.spin_speech_pad.setValue(cfg.speech_pad_ms)
        self.spin_min_speech.setValue(cfg.min_speech_duration_ms)
        self.spin_pre_buffer.setValue(cfg.vad_pre_buffer_ms)
        self.chk_tts_gpu.setChecked(cfg.tts_gpu)

    def write_to_config(self, cfg: Config) -> Config:
        cfg.language = self.combo_language.currentData()
        cfg.whisper_model = self.combo_whisper.currentText()
        cfg.llm_model_path = self.edit_llm_path.text()
        cfg.tts_model = self.combo_tts.currentText()
        cfg.llm_temperature = self.spin_temperature.value()
        cfg.llm_max_tokens = self.spin_max_tokens.value()
        cfg.llm_n_ctx = self.spin_ctx.value()
        cfg.llm_n_gpu_layers = self.spin_gpu_layers.value()
        cfg.max_conversation_turns = self.spin_turns.value()
        cfg.llm_system_prompt = self.edit_system_prompt.toPlainText()
        cfg.vad_threshold = self.spin_vad_threshold.value()
        cfg.min_silence_duration_ms = self.spin_silence_ms.value()
        cfg.speech_pad_ms = self.spin_speech_pad.value()
        cfg.min_speech_duration_ms = self.spin_min_speech.value()
        cfg.vad_pre_buffer_ms = self.spin_pre_buffer.value()
        cfg.tts_gpu = self.chk_tts_gpu.isChecked()
        return cfg

    def _browse_llm(self) -> None:
        start_dir = str(Path(self.edit_llm_path.text()).parent)
        if not Path(start_dir).is_dir():
            start_dir = "models"
        path, _ = QFileDialog.getOpenFileName(
            self, "Valitse GGUF-malli", start_dir, "GGUF-mallit (*.gguf);;Kaikki (*)"
        )
        if path:
            self.edit_llm_path.setText(path)


# ── Main window ────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    """Top-level ROS 2 GUI window.

    Adds a text-input row (compared to the standalone GUI) so the user
    can type messages that are published to ``/voice_chatbot/user_text``
    even when no microphone is available.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Äänichatbot (ROS 2)")
        self.resize(1100, 720)

        self._config = Config.load()
        self._bridge = RosBridge(self)

        self._build_ui()
        self._connect_signals()
        self._set_connected_state(False)

    # ── UI construction ──

    def _build_ui(self) -> None:
        # ── Toolbar ──
        toolbar = self.addToolBar("Hallinta")
        toolbar.setMovable(False)

        self.btn_connect = QPushButton("  Yhdistä")
        self.btn_connect.setMinimumHeight(32)
        self.btn_connect.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-weight: bold; padding: 4px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #388e3c; }"
            "QPushButton:disabled { background-color: #666; }"
        )
        toolbar.addWidget(self.btn_connect)

        self.btn_disconnect = QPushButton("  Katkaise")
        self.btn_disconnect.setMinimumHeight(32)
        self.btn_disconnect.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; "
            "font-weight: bold; padding: 4px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #d32f2f; }"
            "QPushButton:disabled { background-color: #666; }"
        )
        toolbar.addWidget(self.btn_disconnect)

        self.btn_save = QPushButton("  Tallenna asetukset")
        self.btn_save.setMinimumHeight(32)
        self.btn_save.setToolTip("Tallenna asetukset config.json-tiedostoon (voimaan nodien uudelleenkäynnistyksellä)")
        self.btn_save.setStyleSheet(
            "QPushButton { background-color: #1565c0; color: white; "
            "font-weight: bold; padding: 4px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1976d2; }"
        )
        toolbar.addWidget(self.btn_save)

        self.btn_clear = QPushButton("  Tyhjennä keskustelu")
        self.btn_clear.setMinimumHeight(32)
        self.btn_clear.setStyleSheet(
            "QPushButton { padding: 4px 12px; border-radius: 4px; }"
        )
        toolbar.addWidget(self.btn_clear)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.label_status = QLabel("Tila: Ei yhdistetty")
        self.label_status.setTextFormat(Qt.TextFormat.RichText)
        self.label_status.setStyleSheet("font-weight: bold; padding-right: 12px;")
        toolbar.addWidget(self.label_status)

        # ── Central area ──
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Settings panel (left)
        self.settings_panel = SettingsPanel(self._config)
        main_layout.addWidget(self.settings_panel)

        # Right side: chat + text input + log
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setChildrenCollapsible(False)

        # Chat panel
        chat_container = QWidget()
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_label = QLabel("Keskustelu")
        chat_label.setStyleSheet("font-weight: bold; font-size: 13px; padding: 2px;")
        chat_layout.addWidget(chat_label)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Segoe UI", 11))
        self.chat_display.setStyleSheet(
            "QTextEdit { background-color: #1e1e2e; color: #cdd6f4; "
            "border: 1px solid #45475a; border-radius: 4px; padding: 8px; }"
        )
        chat_layout.addWidget(self.chat_display)

        # Text input row
        input_row = QHBoxLayout()
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Kirjoita viesti...")
        self.text_input.setMinimumHeight(32)
        self.text_input.setFont(QFont("Segoe UI", 11))
        self.btn_send = QPushButton("Lähetä")
        self.btn_send.setMinimumHeight(32)
        self.btn_send.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-weight: bold; padding: 4px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #388e3c; }"
            "QPushButton:disabled { background-color: #666; }"
        )
        input_row.addWidget(self.text_input)
        input_row.addWidget(self.btn_send)
        chat_layout.addLayout(input_row)

        right_splitter.addWidget(chat_container)

        # Log panel
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_label = QLabel("Järjestelmäloki")
        log_label.setStyleSheet("font-weight: bold; font-size: 13px; padding: 2px;")
        log_layout.addWidget(log_label)

        self.log_display = QPlainTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Cascadia Mono, Consolas, Courier New", 9))
        self.log_display.setMaximumBlockCount(2000)
        self.log_display.setStyleSheet(
            "QPlainTextEdit { background-color: #11111b; color: #a6adc8; "
            "border: 1px solid #45475a; border-radius: 4px; padding: 4px; }"
        )
        log_layout.addWidget(self.log_display)
        right_splitter.addWidget(log_container)

        right_splitter.setSizes([480, 200])
        main_layout.addWidget(right_splitter, stretch=1)

        self.statusBar().showMessage("Ei yhdistetty")

    # ── Signal wiring ──

    def _connect_signals(self) -> None:
        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_disconnect.clicked.connect(self._on_disconnect)
        self.btn_save.clicked.connect(self._on_save_settings)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_send.clicked.connect(self._on_send)
        self.text_input.returnPressed.connect(self._on_send)

        self._bridge.log_received.connect(
            self._append_log, Qt.ConnectionType.QueuedConnection
        )
        self._bridge.status_received.connect(
            self._update_status, Qt.ConnectionType.QueuedConnection
        )
        self._bridge.chat_message.connect(
            self._append_chat, Qt.ConnectionType.QueuedConnection
        )

    # ── State helpers ──

    def _set_connected_state(self, connected: bool) -> None:
        self.btn_connect.setEnabled(not connected)
        self.btn_disconnect.setEnabled(connected)
        self.btn_send.setEnabled(connected)
        self.text_input.setEnabled(connected)

    # ── Slots ──

    def _on_connect(self) -> None:
        try:
            self._bridge.connect_ros()
            self._set_connected_state(True)
            self._update_status("Yhdistetty")
            self._append_log("ROS 2 -yhteys muodostettu.")
        except Exception as exc:
            self._append_log(f"VIRHE: ROS 2 -yhteys epäonnistui: {exc}")

    def _on_disconnect(self) -> None:
        self._bridge.disconnect_ros()
        self._set_connected_state(False)
        self._update_status("Ei yhdistetty")
        self._append_log("ROS 2 -yhteys katkaistu.")

    def _on_save_settings(self) -> None:
        cfg = Config()
        self.settings_panel.write_to_config(cfg)
        cfg.save()
        self._append_log("Asetukset tallennettu config.json-tiedostoon.")

    def _on_clear(self) -> None:
        self.chat_display.clear()
        self._bridge.clear_history()

    def _on_send(self) -> None:
        text = self.text_input.text().strip()
        if not text:
            return
        self.text_input.clear()
        self._bridge.send_text(text)

    # ── UI update helpers ──

    def _append_log(self, text: str) -> None:
        self.log_display.appendPlainText(text)
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_display.setTextCursor(cursor)

    def _append_chat(self, role: str, text: str) -> None:
        if role == "user":
            color = "#89b4fa"
            prefix = "Sinä"
        else:
            color = "#a6e3a1"
            prefix = "Botti"

        html = (
            f'<p style="margin: 6px 0;">'
            f'<span style="color: {color}; font-weight: bold;">{prefix}:</span> '
            f'<span style="color: #cdd6f4;">{_escape_html(text)}</span>'
            f"</p>"
        )
        self.chat_display.append(html)
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)

    def _update_status(self, text: str) -> None:
        if "Kuunnellaan" in text or "Yhdistetty" in text:
            indicator = '<span style="color: #a6e3a1;">&#9679;</span>'
        elif "Käsitellään" in text or "vastaa" in text or "Puhutaan" in text:
            indicator = '<span style="color: #f9e2af;">&#9679;</span>'
        elif "Puhe havaittu" in text:
            indicator = '<span style="color: #89b4fa;">&#9679;</span>'
        elif "Virhe" in text:
            indicator = '<span style="color: #f38ba8;">&#9679;</span>'
        else:
            indicator = '<span style="color: #6c7086;">&#9679;</span>'

        self.label_status.setText(f"  Tila: {text} {indicator}")
        self.statusBar().showMessage(text)

    # ── Window close ──

    def closeEvent(self, event) -> None:
        self._bridge.disconnect_ros()
        event.accept()


# ── Utility functions ──────────────────────────────────────────────


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def _set_combo_by_data(combo: QComboBox, data) -> None:
    for i in range(combo.count()):
        if combo.itemData(i) == data:
            combo.setCurrentIndex(i)
            return


def _set_combo_by_text(combo: QComboBox, text: str) -> None:
    idx = combo.findText(text)
    if idx >= 0:
        combo.setCurrentIndex(idx)
    elif combo.isEditable():
        combo.setEditText(text)


# ── Entry point ────────────────────────────────────────────────────


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(
        """
        QMainWindow, QWidget { font-family: 'Segoe UI', sans-serif; }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #555;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 14px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
        }
        QToolBar { spacing: 6px; padding: 4px; }
        """
    )

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
