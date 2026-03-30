import json

from voice_chatbot.config import CONFIG_ENV_VAR, Config, default_config_path


def test_save_and_load_round_trip(tmp_path):
    config = Config(
        language="en", llm_temperature=0.25, llm_model_path="models/test.gguf"
    )
    path = tmp_path / "config.json"

    config.save(path)

    loaded = Config.load(path)
    assert loaded == config


def test_load_ignores_unknown_fields(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "language": "sv",
                "llm_max_tokens": 512,
                "unknown_field": "ignored",
            }
        ),
        encoding="utf-8",
    )

    loaded = Config.load(path)

    assert loaded.language == "sv"
    assert loaded.llm_max_tokens == 512
    assert not hasattr(loaded, "unknown_field")


def test_load_returns_defaults_for_missing_file(tmp_path):
    loaded = Config.load(tmp_path / "missing.json")
    assert loaded == Config()


def test_load_returns_defaults_for_invalid_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not valid json", encoding="utf-8")

    loaded = Config.load(path)

    assert loaded == Config()


def test_default_config_path_uses_env_override(monkeypatch, tmp_path):
    target = tmp_path / "custom" / "config.json"
    monkeypatch.setenv(CONFIG_ENV_VAR, str(target))

    assert default_config_path() == target


def test_save_creates_parent_directory_for_default_path(monkeypatch, tmp_path):
    target = tmp_path / "nested" / "config.json"
    monkeypatch.setenv(CONFIG_ENV_VAR, str(target))

    Config(language="de").save()

    assert Config.load() == Config(language="de")
