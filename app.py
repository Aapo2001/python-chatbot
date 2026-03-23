"""
Voice Chatbot - Desktop GUI Application
PySide6 interface with model settings, chat display, and system log.
"""

import os
import sys
import traceback
from pathlib import Path

# ── CUDA DLL setup (must run before any CUDA-dependent imports) ────
_cuda_path = os.environ.get("CUDA_PATH", r"D:\cuda")
for _p in [os.path.join(_cuda_path, "bin", "x64"), os.path.join(_cuda_path, "bin")]:
    if os.path.isdir(_p):
        os.add_dll_directory(_p)
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QFileDialog,
    QGroupBox,
    QScrollArea,
    QPlainTextEdit,
    QTextEdit,
    QFormLayout,
    QCheckBox,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer
from PySide6.QtGui import QFont, QTextCursor, QColor, QIcon

from config import Config

# ── Constants ──────────────────────────────────────────────────────

WHISPER_MODELS = [
    "tiny", "tiny.en", "base", "base.en",
    "small", "small.en", "medium", "medium.en",
    "large-v1", "large-v2", "large-v3",
]

LANGUAGES = {
    "Suomi (fi)": "fi",
    "English (en)": "en",
    "Auto": "auto",
    "Svenska (sv)": "sv",
    "Deutsch (de)": "de",
    "Francais (fr)": "fr",
    "Espanol (es)": "es",
    "Italiano (it)": "it",
    "Portugues (pt)": "pt",
    "Nederlands (nl)": "nl",
    "Polski (pl)": "pl",
    "Japanese (ja)": "ja",
    "Chinese (zh)": "zh",
    "Korean (ko)": "ko",
    "Russian (ru)": "ru",
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


# ── Stdout / stderr redirect → Qt signal ──────────────────────────


class LogStream(QObject):
    """Captures writes to sys.stdout / sys.stderr and emits them as signals."""

    message = Signal(str)

    def write(self, text: str) -> None:
        if text and text.strip():
            self.message.emit(text.rstrip("\n"))

    def flush(self) -> None:
        pass


# ── Background worker ─────────────────────────────────────────────


class ChatbotWorker(QThread):
    """Runs model loading + audio loop in a background thread."""

    log = Signal(str)
    chat_message = Signal(str, str)  # (role, text)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    models_ready = Signal()

    def __init__(self, config: Config, parent: QObject | None = None):
        super().__init__(parent)
        self._config = config
        self._running = False

    # ── public ──

    def stop(self) -> None:
        self._running = False

    # ── thread entry ──

    def run(self) -> None:  # noqa: C901 (complexity ok for worker)
        self._running = True

        # Redirect stdout/stderr so library prints appear in the log panel
        log_stream = LogStream()
        log_stream.message.connect(self.log.emit, Qt.ConnectionType.QueuedConnection)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = log_stream
        sys.stderr = log_stream

        audio = None
        try:
            self._emit_system_info()

            # ── Load models ──
            self.status_changed.emit("Ladataan malleja...")

            self.log.emit("[Init] Audio I/O...")
            from audio_io import AudioIO
            audio = AudioIO(self._config)

            self.log.emit("[Init] Silero-VAD...")
            from vad import VoiceActivityDetector
            vad = VoiceActivityDetector(self._config)

            self.log.emit("[Init] Whisper STT...")
            from stt import SpeechToText
            stt = SpeechToText(self._config)

            self.log.emit("[Init] LLM...")
            from llm import ChatLLM
            llm = ChatLLM(self._config)

            self.log.emit("[Init] Coqui TTS...")
            from tts_engine import TextToSpeech
            tts = TextToSpeech(self._config)

            self.log.emit("Kaikki mallit ladattu onnistuneesti!")
            self.models_ready.emit()
            self.status_changed.emit("Kuunnellaan...")

            # ── Audio loop ──
            audio.start_capture()

            while self._running:
                chunk = audio.get_audio_chunk(timeout=0.1)
                if chunk is None:
                    continue

                event, audio_data = vad.process_chunk(chunk)

                if event == "speech_start":
                    self.status_changed.emit("Puhe havaittu...")

                elif event == "speech_end":
                    self.status_changed.emit("Käsitellään...")

                    text = stt.transcribe(audio_data)
                    if not text or text.isspace():
                        self.status_changed.emit("Kuunnellaan...")
                        continue

                    self.chat_message.emit("user", text)

                    self.status_changed.emit("LLM vastaa...")
                    response = llm.chat(text)
                    self.chat_message.emit("assistant", response)

                    self.status_changed.emit("Puhutaan...")
                    audio_out, sr = tts.synthesize(response)
                    audio.play_audio(audio_out, sr)

                    audio.clear_queue()
                    vad.reset()
                    self.status_changed.emit("Kuunnellaan...")

        except Exception:
            self.error_occurred.emit(traceback.format_exc())

        finally:
            if audio is not None:
                audio.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.status_changed.emit("Pysäytetty")

    # ── helpers ──

    def _emit_system_info(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                self.log.emit(f"GPU: {name} ({vram:.1f} GB)")
                self.log.emit(f"CUDA: {torch.version.cuda}")
            else:
                self.log.emit("VAROITUS: CUDA ei saatavilla – mallit ajetaan CPU:lla.")
        except ImportError:
            self.log.emit("VAROITUS: PyTorch ei asennettu.")


# ── Settings panel ─────────────────────────────────────────────────


class SettingsPanel(QScrollArea):
    """Left sidebar with all editable settings."""

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
        self.spin_pre_buffer.setToolTip(
            "Ääntä tallennetaan ennen puheen tunnistusta, "
            "jotta ensimmäiset tavut eivät leikkaudu pois."
        )
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

        # ── Populate from config ──
        self.load_from_config(config)

    # ── config ↔ widgets ──

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

    # ── slots ──

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
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Äänichatbot")
        self.resize(1100, 720)

        self._config = Config.load()
        self._worker: ChatbotWorker | None = None
        self._pending_restart = False

        self._build_ui()
        self._connect_signals()
        self._set_running_state(False)

    # ── UI construction ──

    def _build_ui(self) -> None:
        # ── Toolbar ──
        toolbar = self.addToolBar("Hallinta")
        toolbar.setMovable(False)
        toolbar.setIconSize(toolbar.iconSize())

        self.btn_start = QPushButton("  Käynnistä")
        self.btn_start.setMinimumHeight(32)
        self.btn_start.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-weight: bold; padding: 4px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #388e3c; }"
            "QPushButton:disabled { background-color: #666; }"
        )
        toolbar.addWidget(self.btn_start)

        self.btn_stop = QPushButton("  Pysäytä")
        self.btn_stop.setMinimumHeight(32)
        self.btn_stop.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; "
            "font-weight: bold; padding: 4px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #d32f2f; }"
            "QPushButton:disabled { background-color: #666; }"
        )
        toolbar.addWidget(self.btn_stop)

        self.btn_restart = QPushButton("  Käynnistä uudelleen")
        self.btn_restart.setMinimumHeight(32)
        self.btn_restart.setToolTip("Pysäytä ja käynnistä uudelleen uusilla asetuksilla")
        self.btn_restart.setStyleSheet(
            "QPushButton { background-color: #e65100; color: white; "
            "font-weight: bold; padding: 4px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #ef6c00; }"
            "QPushButton:disabled { background-color: #666; }"
        )
        toolbar.addWidget(self.btn_restart)

        self.btn_clear = QPushButton("  Tyhjennä keskustelu")
        self.btn_clear.setMinimumHeight(32)
        self.btn_clear.setStyleSheet(
            "QPushButton { padding: 4px 12px; border-radius: 4px; }"
        )
        toolbar.addWidget(self.btn_clear)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.label_status = QLabel("Tila: Valmis")
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

        # Right side: chat + log
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

        # ── Status bar ──
        self.statusBar().showMessage("Valmis")

    # ── Signal wiring ──

    def _connect_signals(self) -> None:
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_restart.clicked.connect(self._on_restart)
        self.btn_clear.clicked.connect(self._on_clear)

    # ── State helpers ──

    def _set_running_state(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_restart.setEnabled(running)
        # Settings always editable — use Restart to apply while running
        self.settings_panel.setEnabled(True)

    def _build_config_from_ui(self) -> Config:
        """Build a fresh Config from current UI widget values."""
        cfg = Config()
        self.settings_panel.write_to_config(cfg)
        cfg.save()
        return cfg

    # ── Slots ──

    def _on_start(self) -> None:
        # Ensure any previous worker is fully stopped
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(5000)
            self._worker = None

        # Build config directly from current UI values (not from config.json)
        self._config = self._build_config_from_ui()

        # Validate LLM path
        if not Path(self._config.llm_model_path).exists():
            self._append_log(
                f"VIRHE: LLM-mallia ei löydy: {self._config.llm_model_path}"
            )
            self._append_log("Suorita ensin: python setup_models.py")
            return

        self._set_running_state(True)
        self.log_display.clear()
        self._append_log("Käynnistetään...")

        self._worker = ChatbotWorker(self._config)
        self._worker.log.connect(self._append_log, Qt.ConnectionType.QueuedConnection)
        self._worker.chat_message.connect(
            self._append_chat, Qt.ConnectionType.QueuedConnection
        )
        self._worker.status_changed.connect(
            self._update_status, Qt.ConnectionType.QueuedConnection
        )
        self._worker.error_occurred.connect(
            self._on_error, Qt.ConnectionType.QueuedConnection
        )
        self._worker.models_ready.connect(
            lambda: self._append_log("Kaikki mallit valmiina."),
            Qt.ConnectionType.QueuedConnection,
        )
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._append_log("Pysäytetään...")
            self._worker.stop()
            self.btn_stop.setEnabled(False)
            self.btn_restart.setEnabled(False)

    def _on_restart(self) -> None:
        """Stop current worker and start a new one with current UI settings."""
        self._append_log("Käynnistetään uudelleen uusilla asetuksilla...")
        self._pending_restart = True
        if self._worker is not None:
            self._worker.stop()
        else:
            self._on_start()

    def _on_clear(self) -> None:
        self.chat_display.clear()

    def _on_worker_finished(self) -> None:
        self._worker = None

        # If a restart was requested, start again immediately
        if getattr(self, "_pending_restart", False):
            self._pending_restart = False
            self._on_start()
            return

        self._set_running_state(False)
        self._update_status("Pysäytetty")
        self._append_log("Chatbot pysäytetty.")

    def _on_error(self, error_text: str) -> None:
        self._append_log(f"VIRHE:\n{error_text}")
        self._update_status("Virhe")

    # ── UI update helpers ──

    def _append_log(self, text: str) -> None:
        self.log_display.appendPlainText(text)
        # Auto-scroll to bottom
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_display.setTextCursor(cursor)

    def _append_chat(self, role: str, text: str) -> None:
        if role == "user":
            color = "#89b4fa"  # blue
            prefix = "Sinä"
        else:
            color = "#a6e3a1"  # green
            prefix = "Botti"

        html = (
            f'<p style="margin: 6px 0;">'
            f'<span style="color: {color}; font-weight: bold;">{prefix}:</span> '
            f'<span style="color: #cdd6f4;">{_escape_html(text)}</span>'
            f"</p>"
        )
        self.chat_display.append(html)

        # Auto-scroll
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)

    def _update_status(self, text: str) -> None:
        # Pick an indicator color
        if "Kuunnellaan" in text:
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
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(5000)
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

    # Apply a subtle global stylesheet
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
