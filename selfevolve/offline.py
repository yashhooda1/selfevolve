"""Offline enforcement.

Two things happen here.

1. `harden_env()` turns off every phone-home we ship near. It runs on import of
   the package, before the libraries that read these variables get a chance to.
   Setting them in your shell would also work; doing it in code means it's true
   even when someone runs a script directly, and it's auditable in one place.

2. `airgap()` installs a socket guard that raises on any connection to a
   non-loopback address. This is the difference between believing the system is
   offline and knowing it: the test suite runs the entire loop under this guard,
   so an accidental `requests.get` in a future edit fails the build rather than
   silently working on a developer's connected laptop.

The guard allows loopback because Ollama is a local service on 127.0.0.1. That
is the one and only permitted destination, and `airgap(allow_loopback=False)`
removes even that.
"""

from __future__ import annotations

import contextlib
import ipaddress
import os
import socket

# Every one of these is a real egress path in a dependency we use or sit next to.
_ENV_LOCKDOWN = {
    # Streamlit pings for usage stats and version checks
    "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    "STREAMLIT_GLOBAL_SHOW_WARNING_ON_DIRECT_EXECUTION": "false",
    # LangChain/LangGraph will ship traces to LangSmith if a key is present in
    # the environment — including one you exported months ago for another project
    "LANGCHAIN_TRACING_V2": "false",
    "LANGSMITH_TRACING": "false",
    "LANGCHAIN_TRACING": "false",
    # HuggingFace: refuse network even if some transitive import wants weights
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    # Assorted telemetry from the wider ecosystem
    "ANONYMIZED_TELEMETRY": "False",
    "CHROMA_TELEMETRY_ENABLED": "False",
    "DO_NOT_TRACK": "1",
    "SCARF_NO_ANALYTICS": "true",
    "POSTHOG_DISABLED": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

# Keys that, if left set, cause a library to try to reach a vendor.
_STRIP_IF_PRESENT = ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_ENDPOINT")


def harden_env() -> None:
    """Idempotent. Never overrides a value the user set on purpose, except the
    tracing keys, which exist only to send data somewhere."""
    for key, value in _ENV_LOCKDOWN.items():
        os.environ.setdefault(key, value)
    for key in _STRIP_IF_PRESENT:
        os.environ.pop(key, None)


def _is_loopback(host: str) -> bool:
    if host in ("localhost", "localhost.localdomain", "", None):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class NetworkBlocked(RuntimeError):
    """Raised when code attempts to leave the machine."""


_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_original_create_connection = socket.create_connection
_installed = False


def install_socket_guard(allow_loopback: bool = True) -> None:
    """Patch socket so any non-loopback connect raises NetworkBlocked."""
    global _installed
    if _installed:
        return

    def _check(address):
        host = address[0] if isinstance(address, tuple) else str(address)
        if allow_loopback and _is_loopback(str(host)):
            return
        raise NetworkBlocked(
            f"selfevolve airgap: blocked outbound connection to {host!r}. "
            "This build is offline by design — nothing here should leave the machine."
        )

    def connect(self, address):
        _check(address)
        return _original_connect(self, address)

    def connect_ex(self, address):
        _check(address)
        return _original_connect_ex(self, address)

    def create_connection(address, *args, **kwargs):
        _check(address)
        return _original_create_connection(address, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.create_connection = create_connection
    _installed = True


def remove_socket_guard() -> None:
    global _installed
    socket.socket.connect = _original_connect
    socket.socket.connect_ex = _original_connect_ex
    socket.create_connection = _original_create_connection
    _installed = False


@contextlib.contextmanager
def airgap(allow_loopback: bool = True):
    """Context manager form, for tests and for `selfevolve doctor`."""
    install_socket_guard(allow_loopback=allow_loopback)
    try:
        yield
    finally:
        remove_socket_guard()


def report() -> dict[str, object]:
    """What `selfevolve doctor` prints. Facts, not reassurance."""
    from .config import Config  # local import keeps module import cheap

    cfg = Config()
    return {
        "data_dir": str(cfg.data_dir.resolve()),
        "db": str(cfg.db_path()),
        "llm_backend": cfg.llm_backend,
        "llm_model": cfg.llm_model,
        "embed_backend": cfg.embed_backend,
        "embed_model": cfg.embed_model,
        "ollama_host": cfg.ollama_host,
        "ollama_is_local": _is_loopback(_host_of(cfg.ollama_host)),
        "env_locked": {k: os.environ.get(k) for k in sorted(_ENV_LOCKDOWN)},
        "leaked_keys": [k for k in _STRIP_IF_PRESENT if os.environ.get(k)],
    }


def _host_of(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).hostname or ""
