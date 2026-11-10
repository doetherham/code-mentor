from typing import Callable
import os

from ..config.settings import Settings

os.environ["NO_PROXY"] = "*"

def build_tools(cfg: Settings) -> dict[str, Callable]:
    if not (cfg.TOOLS_ENABLED and cfg.FILE_ACCESS_ENABLED):
        return {}

    root = cfg.ROOT.resolve()
    skip = set(cfg.SKIP)

    def _safe(path: str):
        """Путь внутри ROOT, выйти за его пределы (../) нельзя."""
        p = (root / path).resolve()
        if not p.is_relative_to(root):
            raise ValueError("путь вне папки проекта")
        return p

    def list_files(path: str = ".") -> str:
        """Показать файлы и папки в директории проекта.

        Args:
            path: относительный путь к папке, "." означает корень проекта.
        """
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
        """Прочитать текстовый файл проекта.

        Args:
            path: относительный путь к файлу.
        """
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

    return {f.__name__: f for f in (list_files, read_file)}