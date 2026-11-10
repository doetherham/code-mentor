from .base import LLMClient


def make_client(cfg) -> LLMClient:
    if cfg.LLM_PROVIDER == "ollama":
        from .ollama.ollama_client import OllamaClient
        return OllamaClient(cfg)
    if cfg.LLM_PROVIDER == "anthropic":
        from .anthropic.anthropic_client import AnthropicClient
        return AnthropicClient(cfg)
    if cfg.LLM_PROVIDER == "openai":
        from .openai.openai_client import OpenAIClient
        return OpenAIClient(cfg)
    raise ValueError(f"Неизвестный провайдер: {cfg.LLM_PROVIDER}")
