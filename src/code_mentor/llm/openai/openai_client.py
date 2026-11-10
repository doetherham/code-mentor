import json

from openai import OpenAI

from ..base import Reply, ToolCall


class OpenAIClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = OpenAI(
            api_key=cfg.API_KEY.get_secret_value() if cfg.API_KEY else None,
            base_url=cfg.BASE_URL,
            timeout=cfg.TIMEOUT,
        )

    def ensure_ready(self) -> None:
        if not self.cfg.API_KEY:
            raise RuntimeError("Не задан API_KEY в .env")

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        kwargs = dict(
            model=self.cfg.MODEL,
            messages=self._convert(system, messages),
            temperature=self.cfg.TEMPERATURE,
            top_p=self.cfg.TOP_P,
            max_tokens=self.cfg.MAX_TOKENS,
        )
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                }}
                for t in tools
            ]

        msg = self.client.chat.completions.create(**kwargs).choices[0].message

        calls = [
            ToolCall(c.id, c.function.name, json.loads(c.function.arguments or "{}"))
            for c in (msg.tool_calls or [])
        ]
        return Reply(msg.content or "", calls)

    @staticmethod
    def _convert(system, messages):
        out = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                            "content": m["content"]})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                out.append({
                    "role": "assistant",
                    "content": m["content"] or None,
                    "tool_calls": [
                        {"id": c.id, "type": "function",
                         "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                        for c in m["tool_calls"]
                    ],
                })
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out