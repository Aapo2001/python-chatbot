import json

from voice_chatbot.config import Config


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "config.json"
    original = Config(language="sv", whisper_model="small", llm_temperature=0.25)

    original.save(path)
    loaded = Config.load(path)

    assert loaded == original


def test_load_ignores_unknown_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "language": "en",
                "whisper_n_threads": 8,
                "unknown_field": "ignored",
            }
        ),
        encoding="utf-8",
    )

    loaded = Config.load(path)

    assert loaded.language == "en"
    assert loaded.whisper_n_threads == 8
    assert not hasattr(loaded, "unknown_field")


def test_load_returns_defaults_when_file_is_missing(tmp_path):
    loaded = Config.load(tmp_path / "missing.json")

    assert loaded == Config()


def test_load_returns_defaults_when_json_is_invalid(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not-valid-json", encoding="utf-8")

    loaded = Config.load(path)

    assert loaded == Config()
