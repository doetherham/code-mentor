from .client import LLMClient
from .ollama.ollama_client import OllamaClient

def get_client(cfg) -> LLMClient:
    return OllamaClient(cfg)