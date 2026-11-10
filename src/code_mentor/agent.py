import ollama

from .config.settings import Settings, get_settings
from .parsing import parse_text_call


def _role(m) -> str:
    return m["role"] if isinstance(m, dict) else m.role


class Agent:
    def __init__(self, tools, history=None, config: Settings | None = None):
        self.config = config or get_settings()
        self.tools = tools
        self.history = history if history is not None else []
        self.client = ollama.Client(
            host=self.config.BASE_URL,
            timeout=self.config.TIMEOUT,
            headers=self._headers(),
        )

    def _headers(self):
        key = self.config.API_KEY
        return {"Authorization": f"Bearer {key.get_secret_value()}"} if key else None


    def ask(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})

        for _ in range(self.config.MAX_STEPS):
            message = self._chat()
            self.history.append(message)

            calls = self._extract_calls(message)
            if not calls:
                return message.content

            for name, args in calls:
                self.history.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": self._run_tool(name, args),
                })

        return "Достигнут лимит шагов"


    def _messages(self) -> list:
        """System-промпт + последние MAX_HISTORY сообщений."""
        recent = self.history[-self.config.MAX_HISTORY:]
        while recent and _role(recent[0]) == "tool":
            recent = recent[1:]
        system = [{"role": "system", "content": self.config.SYSTEM}] if self.config.SYSTEM else []
        return system + recent

    def _chat(self):
        resp = self.client.chat(
            model=self.config.MODEL,
            messages=self._messages(),
            tools=list(self.tools.values()) or None,
            options={
                "temperature": self.config.TEMPERATURE,
                "top_p": self.config.TOP_P,
                "num_predict": self.config.MAX_TOKENS,
            },
        )
        return resp.message


    def _extract_calls(self, message) -> list[tuple[str, dict]]:
        if message.tool_calls:
            return [(c.function.name, dict(c.function.arguments)) for c in message.tool_calls]

        parsed = parse_text_call(message.content, set(self.tools))
        return [parsed] if parsed else []

    def _run_tool(self, name: str, args: dict) -> str:
        fn = self.tools.get(name)
        if fn is None:
            return f"Ошибка: неизвестный инструмент {name}"
        try:
            result = str(fn(**args))
        except TypeError as e:
            return f"Ошибка аргументов: {e}"

        limit = self.config.MAX_TOOL_OUTPUT
        if len(result) > limit:
            result = result[:limit] + "\n...[вывод обрезан]"
        return result