from types import SimpleNamespace

from voice_chatbot.config import Config

from .conftest import import_fresh, install_module, make_module


class DummySignal:
    def connect(self, *args, **kwargs):
        return None

    def emit(self, *args, **kwargs):
        return None


class DummyQThread:
    def __init__(self, parent=None):
        self.parent = parent
        self.finished = DummySignal()

    def wait(self, timeout=None):
        return True


def load_app_module(monkeypatch):
    qt_namespace = SimpleNamespace(
        ConnectionType=SimpleNamespace(QueuedConnection="queued"),
        Orientation=SimpleNamespace(Vertical="vertical"),
        TextFormat=SimpleNamespace(RichText="rich"),
    )
    widget = type("Widget", (), {})

    install_module(
        monkeypatch,
        "voice_chatbot.platform_setup",
        make_module(
            "voice_chatbot.platform_setup",
            setup_cuda=lambda: None,
            setup_pyside6=lambda: None,
            setup_wsl_audio=lambda: None,
        ),
    )
    install_module(
        monkeypatch,
        "PySide6.QtCore",
        make_module(
            "PySide6.QtCore",
            QObject=object,
            Qt=qt_namespace,
            QThread=DummyQThread,
            Signal=lambda *args, **kwargs: DummySignal(),
        ),
    )
    install_module(
        monkeypatch,
        "PySide6.QtGui",
        make_module("PySide6.QtGui", QFont=widget),
    )
    install_module(
        monkeypatch,
        "PySide6.QtWidgets",
        make_module(
            "PySide6.QtWidgets",
            QApplication=widget,
            QHBoxLayout=widget,
            QLabel=widget,
            QLineEdit=widget,
            QMainWindow=widget,
            QPlainTextEdit=widget,
            QPushButton=widget,
            QSizePolicy=SimpleNamespace(
                Policy=SimpleNamespace(Expanding=1, Preferred=2)
            ),
            QSplitter=widget,
            QTextEdit=widget,
            QVBoxLayout=widget,
            QWidget=widget,
        ),
    )
    install_module(
        monkeypatch,
        "voice_chatbot.ui_common",
        make_module(
            "voice_chatbot.ui_common",
            APP_STYLESHEET="",
            SettingsPanel=widget,
            append_chat=lambda *args, **kwargs: None,
            append_log=lambda *args, **kwargs: None,
            update_status_label=lambda *args, **kwargs: None,
        ),
    )
    return import_fresh("voice_chatbot.app")


def test_chatbot_worker_uses_thread_safe_queue_and_toggle(monkeypatch):
    module = load_app_module(monkeypatch)
    worker = module.ChatbotWorker(Config(tts_enabled=False))

    worker.send_text("hei")
    assert worker._text_queue.get_nowait() == "hei"
    assert worker._tts_enabled.is_set() is False

    worker.set_tts_enabled(True)
    assert worker._tts_enabled.is_set() is True


def test_clear_button_clears_display_and_history(monkeypatch):
    module = load_app_module(monkeypatch)
    cleared = {"display": 0, "history": 0}
    window = object.__new__(module.MainWindow)
    window.chat_display = SimpleNamespace(
        clear=lambda: cleared.__setitem__("display", 1)
    )
    window._worker = SimpleNamespace(
        clear_history=lambda: cleared.__setitem__("history", 1)
    )

    module.MainWindow._on_clear(window)

    assert cleared == {"display": 1, "history": 1}
