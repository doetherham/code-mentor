import ollama

class OllamaClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base_url = cfg.BASE_URL
        self.client = ollama.Client(host=self.base_url, timeout=cfg.TIMEOUT)