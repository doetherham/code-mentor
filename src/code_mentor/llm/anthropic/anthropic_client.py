import anthropic

from ..base import Reply, ToolCall


class AnthropicClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = anthropic.Anthropic(
            api_key=cfg.API_KEY.get_secret_value(), timeout=cfg.TIMEOUT
        )

    def ensure_ready(self) -> None:
        if not self.cfg.API_KEY:
            raise RuntimeError("Не задан API_KEY в .env")

    def chat(self, system, messages, tools) -> Reply:
        resp = self.client.messages.create(
            model=self.cfg.MODEL,
            max_tokens=self.cfg.MAX_TOKENS,
            temperature=self.cfg.TEMPERATURE,
            system=system,
            messages=self._convert(messages),
            tools=[
                {"name": t["name"], "description": t["description"],
                 "input_schema": t["parameters"]}
                for t in tools
            ],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, b.input) for b in resp.content if b.type == "tool_use"]
        return Reply(text, calls)

    @staticmethod
    def _convert(messages):
        out = []
        for m in messages:
            if m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                         "content": m["content"]}
                # подряд идущие результаты склеиваются в одно user-сообщение
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                blocks = [{"type": "text", "text": m["content"]}] if m["content"] else []
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}
                           for c in m["tool_calls"]]
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out