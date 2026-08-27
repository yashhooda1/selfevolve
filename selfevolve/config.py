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
    # Instruct-tuned, not reasoning-tuned, and that is deliberate. Under a
    # JSON-schema constraint a reasoning model's thinking phase has nowhere to go:
    # qwen3:8b timed out at 180s on a real review where qwen2.5:7b-instruct
    # finished the same file in 22-53s. Reasoning quality is not the bottleneck
    # here -- the model supplies general code knowledge and the learned rules
    # supply the judgement.
    llm_model: str = _env("SELFEVOLVE_LLM_MODEL", "qwen2.5:7b-instruct")
    ollama_host: str = _env("OLLAMA_HOST", "http://127.0.0.1:11434")
    llm_timeout: int = _env_int("SELFEVOLVE_LLM_TIMEOUT", 180)
    warm_timeout: int = _env_int("SELFEVOLVE_WARM_TIMEOUT", 120)
    num_ctx: int = _env_int("SELFEVOLVE_NUM_CTX", 8192)
    # How much of a file is actually sent for review. A 2,000-line module blows
    # past the context window and turns a review into a several-minute wait for a
    # worse answer; reviewing the first N characters well beats timing out.
    max_input_chars: int = _env_int("SELFEVOLVE_MAX_INPUT_CHARS", 12000)
    # None = don't send the field at all. "false" disables reasoning-model
    # thinking, which is usually what you want under a schema constraint.
    think: bool | None = None

    # --- retrieval ---
    top_k_insights: int = _env_int("SELFEVOLVE_TOP_K_INSIGHTS", 6)
    top_k_trajectories: int = _env_int("SELFEVOLVE_TOP_K_TRAJECTORIES", 2)
    min_confidence: float = _env_float("SELFEVOLVE_MIN_CONFIDENCE", 0.25)
    retire_below: float = _env_float("SELFEVOLVE_RETIRE_BELOW", 0.15)

    def __post_init__(self) -> None:
        raw_think = os.environ.get("SELFEVOLVE_LLM_THINK")
        if raw_think is not None:
            self.think = raw_think.strip().lower() in ("1", "true", "yes", "on")
        # Normalize a bare host into a URL, and tolerate the `host:port` form
        # Ollama's own docs use for OLLAMA_HOST.
        if not self.ollama_host.startswith(("http://", "https://")):
            self.ollama_host = f"http://{self.ollama_host}"
        self.ollama_host = self.ollama_host.rstrip("/")

    def ensure_dir(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir

    def sync_folder_warning(self) -> str | None:
        """Detect a database sitting inside a cloud-sync folder.

        SQLite and file sync are a genuinely bad pair. OneDrive, Dropbox and
        iCloud copy a file whenever it changes, with no idea that a database has
        a journal that must stay consistent with it — and OneDrive on Windows can
        also hold a lock mid-upload, which surfaces as "database is locked" at
        random. The failure is intermittent and looks like a bug in this program.

        This never blocks anything; it prints a warning once so that when it does
        go wrong, you know why.
        """
        parts = {p.lower() for p in self.data_dir.resolve().parts}
        for marker, name in (
            ("onedrive", "OneDrive"),
            ("dropbox", "Dropbox"),
            ("google drive", "Google Drive"),
            ("googledrive", "Google Drive"),
            ("icloud drive", "iCloud Drive"),
            ("com~apple~clouddocs", "iCloud Drive"),
        ):
            if any(marker in p for p in parts):
                return (
                    f"experience.db is inside {name}, which syncs files as they change. "
                    "SQLite does not tolerate that well — expect intermittent "
                    '"database is locked" errors and, rarely, corruption. '
                    "Point SELFEVOLVE_DATA_DIR somewhere outside it:\n"
                    "    setx SELFEVOLVE_DATA_DIR \"%USERPROFILE%\\.selfevolve\"    (Windows)\n"
                    "    export SELFEVOLVE_DATA_DIR=~/.selfevolve                 (macOS/Linux)"
                )
        return None

    def db_path(self) -> Path:
        return self.ensure_dir() / self.db_file

    def checkpoint_path(self) -> Path:
        return self.ensure_dir() / "checkpoints.sqlite"


DEFAULT = Config()
