import json
import re


def parse_text_call(content: str | None, known: set[str]) -> tuple[str, dict] | None:
    """Достаёт вызов инструмента из текста ответа модели.
    Понимает JSON вида {"name": "list_files", "arguments": {"path": "."}},
    в том числе внутри блока ```json ... ```.
    Возвращает (имя, аргументы) или None, если вызова нет.
    """
    if not content:
        return None

    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    name = data.get("name")
    args = data.get("arguments", data.get("parameters", {}))
    if name in known and isinstance(args, dict):
        return name, args
    return None