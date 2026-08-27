"""Embeddings, without ever reaching the internet.

Backends, in preference order:

  ollama   POST http://localhost:11434/api/embeddings with nomic-embed-text.
           You already run Ollama for generation, so this adds no second model
           runtime and no weights to vendor — one `ollama pull` and the machine
           can be offline forever after.

  hash     Deterministic hashed n-grams, pure stdlib. Semantically weak, but it
           needs nothing at all: no model, no service, no download. Used
           automatically when Ollama isn't reachable so the loop degrades instead
           of crashing, and used deliberately in tests for determinism.

Deliberately absent: sentence-transformers. It's a better embedder, but it
fetches weights from huggingface.co on first use, and a build that quietly
downloads 130MB the first time you run it is not offline — it's offline *after*
you've already been online. If you want BGE quality, `ollama pull bge-m3` and set
SELFEVOLVE_EMBED_MODEL.

The store passes in a SQLite-backed cache, so a given string is embedded once
ever, no matter how many times a rule is reinforced.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request

from .config import DEFAULT, Config

_TOKEN = re.compile(r"[a-z0-9_]+")
_HASH_DIM = 384


class Embedder:
    def __init__(self, cfg: Config | None = None, cache=None):
        self.cfg = cfg or DEFAULT
        self.cache = cache
        self.backend = self._resolve_backend()

    def _resolve_backend(self) -> str:
        want = self.cfg.embed_backend
        if want == "hash":
            return "hash"
        if want == "ollama":
            return "ollama"  # explicit request: fail loudly later if unreachable
        return "ollama" if self._ollama_alive() else "hash"

    def _ollama_alive(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.cfg.ollama_host}/api/tags")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            return False

    @property
    def model(self) -> str:
        return self.cfg.embed_model if self.backend == "ollama" else "hash-384"

    def embed_one(self, text: str) -> list[float]:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
        if self.cache is not None:
            hit = self.cache.get(key, self.model)
            if hit:
                return hit
        vec = self._embed_uncached(text)
        # Quantize to float32 here, matching how the cache stores it. Without
        # this, the same text yields float64 on a cache miss and float32 on a
        # hit — tiny differences that make scores depend on cache state, which
        # is the kind of nondeterminism that wastes an afternoon later.
        vec = _to_float32(vec)
        if self.cache is not None and vec:
            self.cache.put(key, self.model, vec)
        return vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def _embed_uncached(self, text: str) -> list[float]:
        if self.backend == "ollama":
            try:
                return _normalize(self._ollama_embed(text))
            except Exception:
                # Ollama died mid-session. Fall back rather than lose the write —
                # a weaker vector beats a dropped lesson. The backend flips for
                # the rest of the process so we don't retry on every call.
                self.backend = "hash"
        return list(_hash_embed(text))

    def _ollama_embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.cfg.embed_model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{self.cfg.ollama_host}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout) as resp:
            body = json.loads(resp.read())
        vec = body.get("embedding") or []
        if not vec:
            raise RuntimeError(f"ollama returned no embedding for model {self.cfg.embed_model}")
        return [float(x) for x in vec]

    @property
    def dim(self) -> int:
        return len(self.embed_one("dimension probe")) or _HASH_DIM


def _to_float32(vec: list[float]) -> list[float]:
    """Round-trip through the same width the store persists, so a value is
    identical whether it came from the cache or was just computed."""
    from array import array

    return list(array("f", vec)) if vec else vec


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _hash_embed(text: str) -> tuple[float, ...]:
    vec = [0.0] * _HASH_DIM
    tokens = _TOKEN.findall(text.lower())
    if not tokens:
        return tuple(vec)
    # unigrams + bigrams, so short rules and code snippets still separate
    grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
    for g in grams:
        h = int(hashlib.md5(g.encode()).hexdigest(), 16)
        vec[h % _HASH_DIM] += 1.0 if (h >> 8) % 2 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return tuple(v / norm for v in vec)


def get_embedder(cfg: Config | None = None, cache=None) -> Embedder:
    return Embedder(cfg or DEFAULT, cache=cache)
