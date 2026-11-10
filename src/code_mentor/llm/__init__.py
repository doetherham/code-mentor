def make_client(cfg):
    if cfg.LLM_PROVIDER == "ollama":
        from .ollama import OllamaClient
        return OllamaClient(cfg)
    if cfg.LLM_PROVIDER == "anthropic":
        from .anthropic.anthropic_client import AnthropicClient
        return AnthropicClient(cfg)
    if cfg.LLM_PROVIDER == "openai":
        from .openai.openai_client import OpenAIClient
        return OpenAIClient(cfg)
    raise ValueError(cfg.LLM_PROVIDER)