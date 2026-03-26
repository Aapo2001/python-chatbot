from voice_chatbot.config import Config

from .conftest import import_fresh, install_module, make_module


class FakeLlama:
    init_calls = []
    responses = []

    def __init__(self, **kwargs):
        type(self).init_calls.append(kwargs)
        self.calls = []
        type(self).instance = self

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        content = type(self).responses.pop(0)
        return {"choices": [{"message": {"content": content}}]}


def load_llm_module(monkeypatch):
    FakeLlama.init_calls = []
    FakeLlama.responses = []
    install_module(monkeypatch, "llama_cpp", make_module("llama_cpp", Llama=FakeLlama))
    install_module(
        monkeypatch,
        "llama_cpp.llama_types",
        make_module(
            "llama_cpp.llama_types",
            ChatCompletionRequestAssistantMessage=dict,
            ChatCompletionRequestMessage=dict,
            ChatCompletionRequestSystemMessage=dict,
            ChatCompletionRequestUserMessage=dict,
            CreateChatCompletionResponse=dict,
        ),
    )
    return import_fresh("voice_chatbot.llm")


def test_initialization_passes_expected_model_settings(monkeypatch):
    module = load_llm_module(monkeypatch)
    config = Config(llm_model_path="models/model.gguf", llm_n_gpu_layers=7, llm_n_ctx=4096)

    module.ChatLLM(config)

    assert FakeLlama.init_calls == [
        {
            "model_path": "models/model.gguf",
            "n_gpu_layers": 7,
            "n_ctx": 4096,
            "verbose": False,
        }
    ]


def test_chat_sends_system_prompt_and_persists_turn(monkeypatch):
    module = load_llm_module(monkeypatch)
    FakeLlama.responses = ["Moi!"]
    chat = module.ChatLLM(Config(llm_system_prompt="system prompt"))

    reply = chat.chat("Hei")

    assert reply == "Moi!"
    messages = FakeLlama.instance.calls[0]["messages"]
    assert messages == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "Hei"},
    ]
    assert chat._conversation_history == [
        {"role": "user", "content": "Hei"},
        {"role": "assistant", "content": "Moi!"},
    ]


def test_chat_returns_empty_string_without_mutating_history(monkeypatch):
    module = load_llm_module(monkeypatch)
    FakeLlama.responses = [""]
    chat = module.ChatLLM(Config())

    reply = chat.chat("Hei")

    assert reply == ""
    assert chat._conversation_history == []


def test_chat_trims_history_to_complete_turns(monkeypatch):
    module = load_llm_module(monkeypatch)
    FakeLlama.responses = ["A1", "A2", "A3"]
    chat = module.ChatLLM(Config(llm_system_prompt="sys", max_conversation_turns=2))

    assert chat.chat("U1") == "A1"
    assert chat.chat("U2") == "A2"
    assert chat.chat("U3") == "A3"

    third_call_messages = FakeLlama.instance.calls[2]["messages"]
    assert third_call_messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "U3"},
    ]
    assert chat._conversation_history == [
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "U3"},
        {"role": "assistant", "content": "A3"},
    ]


def test_clear_history_removes_all_turns(monkeypatch):
    module = load_llm_module(monkeypatch)
    FakeLlama.responses = ["A1"]
    chat = module.ChatLLM(Config())
    chat.chat("U1")

    chat.clear_history()

    assert chat._conversation_history == []
