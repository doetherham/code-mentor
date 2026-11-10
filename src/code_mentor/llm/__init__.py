from .base import LLMClient


def make_client(cfg) -> LLMClient:
    if cfg.LLM_PROVIDER == "ollama":
        from .ollama.ollama_client import OllamaClient
        return OllamaClient(cfg)
