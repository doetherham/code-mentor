from .client import LLMClient
from .ollama.ollama_client import OllamaClient


def make_client(cfg) -> LLMClient:
    return OllamaClient(cfg)