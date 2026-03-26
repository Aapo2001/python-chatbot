from types import SimpleNamespace

from voice_chatbot.config import Config


def _build_llm_module(fresh_import, module_factory):
    state = {"responses": []}

    class FakeLlama:
        def __init__(self, **kwargs):
            state["init_kwargs"] = kwargs

        def create_chat_completion(self, **kwargs):
            state.setdefault("calls", []).append(kwargs)
            content = state["responses"].pop(0)
            return {"choices": [{"message": {"content": content}}]}

    llama_types = module_factory(
        "llama_cpp.llama_types",
        ChatCompletionRequestAssistantMessage=dict,
        ChatCompletionRequestMessage=dict,
        ChatCompletionRequestSystemMessage=dict,
        ChatCompletionRequestUserMessage=dict,
        CreateChatCompletionResponse=dict,
    )
    llama_cpp = module_factory(
        "llama_cpp",
        Llama=FakeLlama,
        llama_types=llama_types,
    )

    module = fresh_import(
        "voice_chatbot.llm",
        stub_modules={
            "llama_cpp": llama_cpp,
            "llama_cpp.llama_types": llama_types,
        },
        clear_modules=["voice_chatbot.llm"],
    )
    return module, state


def test_chat_passes_system_prompt_and_generation_options(fresh_import, module_factory):
    module, state = _build_llm_module(fresh_import, module_factory)
    state["responses"] = ["terve"]
    config = Config(
        llm_model_path="models/test.gguf",
        llm_n_gpu_layers=12,
        llm_n_ctx=4096,
        llm_max_tokens=128,
        llm_temperature=0.2,
        llm_system_prompt="system",
    )

    llm = module.ChatLLM(config)
    reply = llm.chat("hei")

    assert reply == "terve"
    assert state["init_kwargs"] == {
        "model_path": "models/test.gguf",
        "n_gpu_layers": 12,
        "n_ctx": 4096,
        "verbose": False,
    }

    call = state["calls"][0]
    assert call["max_tokens"] == 128
    assert call["temperature"] == 0.2
    assert call["stream"] is False
    assert call["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hei"},
    ]


def test_chat_trims_to_recent_complete_turns(fresh_import, module_factory):
    module, state = _build_llm_module(fresh_import, module_factory)
    state["responses"] = ["v1", "v2", "v3"]
    llm = module.ChatLLM(Config(max_conversation_turns=2, llm_system_prompt="sys"))

    llm.chat("u1")
    llm.chat("u2")
    llm.chat("u3")

    third_call_messages = state["calls"][-1]["messages"]
    assert third_call_messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "v2"},
        {"role": "user", "content": "u3"},
    ]
    assert llm._conversation_history == [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "v2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "v3"},
    ]


def test_empty_response_does_not_append_assistant_message(fresh_import, module_factory):
    module, state = _build_llm_module(fresh_import, module_factory)
    state["responses"] = [None]
    llm = module.ChatLLM(Config())

    reply = llm.chat("hei")

    assert reply == ""
    assert llm._conversation_history == [{"role": "user", "content": "hei"}]


def test_clear_history_removes_all_messages(fresh_import, module_factory):
    module, state = _build_llm_module(fresh_import, module_factory)
    state["responses"] = ["vastaus"]
    llm = module.ChatLLM(Config())
    llm.chat("hei")

    llm.clear_history()

    assert llm._conversation_history == []
