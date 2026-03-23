from llama_cpp import Llama

from config import Config


class ChatLLM:
    def __init__(self, config: Config):
        print(f"[LLM] Loading model from '{config.llm_model_path}'...")
        self._llm = Llama(
            model_path=config.llm_model_path,
            n_gpu_layers=config.llm_n_gpu_layers,
            n_ctx=config.llm_n_ctx,
            verbose=False,
        )
        self._system_prompt = config.llm_system_prompt
        self._max_turns = config.max_conversation_turns
        self._max_tokens = config.llm_max_tokens
        self._temperature = config.llm_temperature
        self._conversation_history: list[dict[str, str]] = []
        print("[LLM] Model loaded.")

    def chat(self, user_message: str) -> str:
        self._conversation_history.append(
            {"role": "user", "content": user_message}
        )

        # Trim oldest turn pairs if exceeding limit
        max_messages = self._max_turns * 2
        if len(self._conversation_history) > max_messages:
            self._conversation_history = self._conversation_history[-max_messages:]

        messages = [
            {"role": "system", "content": self._system_prompt},
            *self._conversation_history,
        ]

        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        assistant_text = response["choices"][0]["message"]["content"]
        self._conversation_history.append(
            {"role": "assistant", "content": assistant_text}
        )

        return assistant_text

    def clear_history(self):
        self._conversation_history = []
