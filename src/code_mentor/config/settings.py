from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from .paths import get_root

ENV_FILE = get_root() / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8-sig",
        env_ignore_empty=True,
        extra="ignore",
    )

    # LLM
    LLM_PROVIDER: Literal["ollama"] = "ollama"
    MODEL: str
    BASE_URL: str = "http://localhost:11434"
    API_KEY: SecretStr | None = None

    # AGENT CONFIG
    TEMPERATURE: float = Field(0.7, ge=0, le=2)
    TOP_P: float = Field(0.9, ge=0, le=1)
    MAX_TOKENS: int = 4096
    TIMEOUT: int = 120
    MAX_STEPS: int = 5

    # BEHAVIOR
    SYSTEM: str = ""

    # CONTEXT AND MEMORY
    MAX_HISTORY: int = 20
    MAX_CHARS: int = 5000
    MAX_TOOL_OUTPUT: int = 10000

    # TOOLS
    TOOLS_ENABLED: bool = True
    WEB_SEARCH_ENABLED: bool = False
    CODE_EXECUTION_ENABLED: bool = False
    FILE_ACCESS_ENABLED: bool = True

    # FILESYSTEM
    SKIP: Annotated[list[str], NoDecode] = []
    ROOT: Path = Field(default_factory=get_root)
    DATA_DIR: Path = Path("data")
    LOG_DIR: Path = Path("logs")

    @field_validator("SKIP", mode="before")
    @classmethod
    def _split_skip(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("ROOT", mode="before")
    @classmethod
    def _default_root(cls, v):
        return get_root() if v == "default" else v

    @model_validator(mode="after")
    def _anchor_dirs(self):
        if not self.DATA_DIR.is_absolute():
            self.DATA_DIR = self.ROOT / self.DATA_DIR
        if not self.LOG_DIR.is_absolute():
            self.LOG_DIR = self.ROOT / self.LOG_DIR
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()