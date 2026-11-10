import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse


def ensure_ollama(client, base_url: str, timeout: float = 30) -> None:
    """Проверяет Ollama и при необходимости запускает локальный сервер."""
    try:
        client.list()
        return
    except Exception:
        pass

    if urlparse(base_url).hostname not in {"localhost", "127.0.0.1"}:
        raise RuntimeError(f"Сервер Ollama недоступен: {base_url}")

    exe = shutil.which("ollama")
    if exe is None:
        raise RuntimeError("Ollama не установлена: https://ollama.com/download")

    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    subprocess.Popen(
        [exe, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            client.list()
            return
        except Exception:
            time.sleep(0.5)

    raise RuntimeError("Ollama не запустилась за отведённое время")