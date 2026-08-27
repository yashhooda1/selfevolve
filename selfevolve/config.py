"""Configuration.

Every default points at something on this machine. There is no value in here
that names a remote host, and `selfevolve doctor` will tell you if you've
overridden one to something that isn't loopback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # --- storage: one SQLite file, no server ---
    data_dir: Path = field(
        default_factory=lambda: Path(_env("SELFEVOLVE_DATA_DIR", "./.selfevolve")).expanduser()
    )
    db_file: str = "experience.db"

    # --- embeddings: local Ollama, or stdlib hashing ---
    embed_backend: str = _env("SELFEVOLVE_EMBED_BACKEND", "auto")  # auto|ollama|hash
    embed_model: str = _env("SELFEVOLVE_EMBED_MODEL", "nomic-embed-text")

    # --- generation: local Ollama ---
    llm_backend: str = _env("SELFEVOLVE_LLM_BACKEND", "ollama")  # ollama|fake
    llm_model: str = _env("SELFEVOLVE_LLM_MODEL", "qwen3:8b")
    ollama_host: str = _env("OLLAMA_HOST", "http://127.0.0.1:11434")
    llm_timeout: int = _env_int("SELFEVOLVE_LLM_TIMEOUT", 180)

    # --- retrieval ---
    top_k_insights: int = _env_int("SELFEVOLVE_TOP_K_INSIGHTS", 6)
    top_k_trajectories: int = _env_int("SELFEVOLVE_TOP_K_TRAJECTORIES", 2)
    min_confidence: float = _env_float("SELFEVOLVE_MIN_CONFIDENCE", 0.25)
    retire_below: float = _env_float("SELFEVOLVE_RETIRE_BELOW", 0.15)

    def __post_init__(self) -> None:
        # Normalize a bare host into a URL, and tolerate the `host:port` form
        # Ollama's own docs use for OLLAMA_HOST.
        if not self.ollama_host.startswith(("http://", "https://")):
            self.ollama_host = f"http://{self.ollama_host}"
        self.ollama_host = self.ollama_host.rstrip("/")

    def ensure_dir(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir

    def db_path(self) -> Path:
        return self.ensure_dir() / self.db_file

    def checkpoint_path(self) -> Path:
        return self.ensure_dir() / "checkpoints.sqlite"


DEFAULT = Config()
