from .config.settings import Settings, get_settings
from .llm.base import LLMClient
from .tools.tools import Tool


class Agent:
    def __init__(self, llm: LLMClient, tools: list[Tool],
                 history: list[dict] | None = None,
                 config: Settings | None = None):
        self.llm = llm
        self.config = config or get_settings()
        self.tools = {t.name: t for t in tools}
        self.schemas = [t.schema() for t in tools]
        self.history = history if history is not None else []

    def ask(self, user_message: str) -> str:
        start = len(self.history)
        self.history.append({"role": "user", "content": user_message})

        try:
            for _ in range(self.config.MAX_STEPS):
                reply = self.llm.chat(self.config.SYSTEM, self._window(), self.schemas)
                self.history.append({
                    "role": "assistant",
                    "content": reply.text,
                    "tool_calls": reply.tool_calls,
                })

                if not reply.tool_calls:
                    return reply.text

                for call in reply.tool_calls:
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": self._run_tool(call.name, call.args),
                    })
        except Exception:
            del self.history[start:]
            raise

        text = "Достигнут лимит шагов"
        self.history.append({"role": "assistant", "content": text})
        return text

    def _window(self) -> list[dict]:
        """Последние MAX_HISTORY сообщений, начиная с сообщения пользователя."""
        recent = self.history[-self.config.MAX_HISTORY:]
        while recent and recent[0]["role"] != "user":
            recent = recent[1:]
        return recent

    def _run_tool(self, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"Ошибка: неизвестный инструмент {name}"
        try:
            result = str(tool.fn(**args))
        except TypeError as e:
            return f"Ошибка аргументов: {e}"

        limit = self.config.MAX_TOOL_OUTPUT
        if len(result) > limit:
            result = result[:limit] + "\n...[вывод обрезан]"
        return result
