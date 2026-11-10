from dataclasses import dataclass
from typing import Callable

from ..config.settings import Settings


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable

    def schema(self) -> dict:
        """Нейтральное описание для LLMClient (каждый адаптер переведёт его сам)."""
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters}


def build_tools(cfg: Settings) -> list[Tool]:
    if not (cfg.TOOLS_ENABLED and cfg.FILE_ACCESS_ENABLED):
        return []

    root = cfg.ROOT.resolve()
    skip = set(cfg.SKIP)

    def _safe(path: str):
        """Путь внутри ROOT, выйти за его пределы (../) нельзя."""
        p = (root / path).resolve()
        if not p.is_relative_to(root):
            raise ValueError("путь вне папки проекта")
        return p

    def list_files(path: str = ".") -> str:
        try:
            d = _safe(path)
            if not d.is_dir():
                return f"Ошибка: {path} не папка"
            items = [
                p.name + ("/" if p.is_dir() else "")
                for p in sorted(d.iterdir())
                if p.name not in skip
            ]
            return "\n".join(items) or "(пусто)"
        except (ValueError, OSError) as e:
            return f"Ошибка: {e}"

    def read_file(path: str) -> str:
        try:
            f = _safe(path)
            if not f.is_file():
                return f"Ошибка: {path} не файл"
            text = f.read_text(encoding="utf-8", errors="replace")
        except (ValueError, OSError) as e:
            return f"Ошибка: {e}"

        if len(text) > cfg.MAX_CHARS:
            return text[: cfg.MAX_CHARS] + f"\n...[обрезано, всего {len(text)} символов]"
        return text

    return [
        Tool(
            name="list_files",
            description="Показать файлы и папки в директории проекта.",
            parameters={
                "type": "object",
                "properties": {"path": {
                    "type": "string",
                    "description": 'Относительный путь к папке, "." означает корень проекта.',
                }},
                "required": [],
            },
            fn=list_files,
        ),
        Tool(
            name="read_file",
            description="Прочитать текстовый файл проекта.",
            parameters={
                "type": "object",
                "properties": {"path": {
                    "type": "string",
                    "description": "Относительный путь к файлу.",
                }},
                "required": ["path"],
            },
            fn=read_file,
        ),
    ]
