from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    def ensure_ready(self) -> None: ...
    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> Reply: ...