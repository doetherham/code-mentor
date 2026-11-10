import os
import uuid

os.environ["NO_PROXY"] = "*"

import ollama

from .run_ollama import ensure_ollama
from ...parsing import parse_text_call
from ..base import Reply, ToolCall

DEFAULT_URL = "http://localhost:11434"


class OllamaClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base_url = cfg.BASE_URL or DEFAULT_URL

        key = cfg.API_KEY.get_secret_value() if cfg.API_KEY else None
        self.client = ollama.Client(
            host=self.base_url,
            timeout=cfg.TIMEOUT,
            headers={"Authorization": f"Bearer {key}"} if key else None,
        )

    def ensure_ready(self) -> None:
        ensure_ollama(self.client, self.base_url)

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        resp = self.client.chat(
            model=self.cfg.MODEL,
            messages=self._convert(system, messages),
            tools=self._schemas(tools) or None,
            options={
                "temperature": self.cfg.TEMPERATURE,
                "top_p": self.cfg.TOP_P,
                "num_predict": self.cfg.MAX_TOKENS,
            },
        )
        msg = resp.message
        text = msg.content or ""

        calls = [
            ToolCall(uuid.uuid4().hex, c.function.name, dict(c.function.arguments))
            for c in (msg.tool_calls or [])
        ]

        if not calls:
            parsed = parse_text_call(text, {t["name"] for t in tools})
            if parsed:
                name, args = parsed
                calls = [ToolCall(uuid.uuid4().hex, name, args)]
                text = ""

        return Reply(text, calls)

    @staticmethod
    def _schemas(tools: list[dict]) -> list[dict]:
        return [
            {"type": "function", "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            }}
            for t in tools
        ]

    @staticmethod
    def _convert(system: str, messages: list[dict]) -> list[dict]:
        out = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m["role"] == "tool":
                out.append({"role": "tool", "tool_name": m["name"],
                            "content": m["content"]})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                out.append({
                    "role": "assistant",
                    "content": m["content"] or "",
                    "tool_calls": [
                        {"function": {"name": c.name, "arguments": c.args}}
                        for c in m["tool_calls"]
                    ],
                })
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out