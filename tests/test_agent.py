import pytest

from code_mentor.agent import Agent
from code_mentor.config.settings import Settings
from code_mentor.llm.base import Reply, ToolCall
from code_mentor.tools.tools import build_tools


class FakeLLM:
    """Отдаёт заготовленные Reply и запоминает, что ему прислали."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def ensure_ready(self):
        pass

    def chat(self, system, messages, tools):
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        return self.replies.pop(0)


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "a.py").write_text("print('hi')")
    (tmp_path / ".git").mkdir()
    return Settings(_env_file=None, MODEL="test", ROOT=tmp_path, MAX_STEPS=3,
                    SYSTEM="sys", SKIP=".git")


def test_plain_answer(cfg):
    llm = FakeLLM([Reply("Привет!")])
    agent = Agent(llm, [], config=cfg)

    assert agent.ask("Привет") == "Привет!"
    assert llm.calls[0]["system"] == "sys"


def test_tool_roundtrip(cfg):
    llm = FakeLLM([
        Reply("", [ToolCall("1", "list_files", {"path": "."})]),
        Reply("Есть a.py"),
    ])
    agent = Agent(llm, build_tools(cfg), config=cfg)

    assert agent.ask("что в папке?") == "Есть a.py"
    tool_msg = next(m for m in agent.history if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "1"
    assert "a.py" in tool_msg["content"]
    assert ".git" not in tool_msg["content"]
    assert {t["name"] for t in llm.calls[0]["tools"]} == {"list_files", "read_file"}


def test_step_limit(cfg):
    llm = FakeLLM([Reply("", [ToolCall(str(i), "list_files", {})]) for i in range(10)])
    agent = Agent(llm, build_tools(cfg), config=cfg)

    assert agent.ask("loop") == "Достигнут лимит шагов"
    assert agent.history[-1]["role"] == "assistant"


def test_error_does_not_pollute_history(cfg):
    class Boom(FakeLLM):
        def chat(self, *a):
            raise ConnectionError

    agent = Agent(Boom([]), [], config=cfg)
    with pytest.raises(ConnectionError):
        agent.ask("hi")
    assert agent.history == []


def test_window_starts_with_user(cfg):
    cfg.MAX_HISTORY = 2
    agent = Agent(FakeLLM([]), [], config=cfg)
    agent.history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [ToolCall("1", "x", {})]},
        {"role": "tool", "tool_call_id": "1", "name": "x", "content": "r"},
    ]
    assert agent._window() == []


def test_path_escape_blocked(cfg):
    read = {t.name: t for t in build_tools(cfg)}["read_file"].fn
    assert "Ошибка" in read("../../etc/passwd")


def test_cloud_provider_needs_key(tmp_path):
    with pytest.raises(Exception):
        Settings(_env_file=None, MODEL="x", LLM_PROVIDER="openai", ROOT=tmp_path)
