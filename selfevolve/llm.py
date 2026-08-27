"""LLM layer with structured output.

Two providers ship:

  OllamaProvider  local models, JSON-schema-constrained decoding via Ollama's
                  `format` parameter (supported since Ollama 0.5). Talks to
                  127.0.0.1 with urllib — no `ollama` package, no `requests`,
                  nothing to audit for phone-home behaviour.
  FakeProvider    deterministic, no service at all. Used by the tests and by
                  `--backend fake` so you can exercise the whole graph — including
                  retrieval, reflection and persistence — before pulling a model.

Swapping in another local runtime (llama.cpp's server, LM Studio, vLLM) means
implementing one method: `structured`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from .config import DEFAULT, Config

T = TypeVar("T", bound=BaseModel)


class LLMProvider(Protocol):
    name: str

    def structured(self, system: str, prompt: str, schema: type[T]) -> T:  # pragma: no cover
        ...


class OllamaError(RuntimeError):
    pass


class OllamaProvider:
    name = "ollama"

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or DEFAULT

    def warm(self) -> None:
        """Load the model before the real request.

        A cold model spends its first seconds mapping several GB into memory,
        and that time is charged against the same timeout as the actual work --
        so a first review can fail for a reason that has nothing to do with the
        review. Warming separately makes the timeout mean what it says.
        """
        payload = {"model": self.cfg.llm_model, "messages": [], "stream": False}
        req = urllib.request.Request(
            f"{self.cfg.ollama_host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.warm_timeout):
                pass
        except Exception:
            pass  # best effort; the real call reports properly if it matters

    def structured(self, system: str, prompt: str, schema: type[T]) -> T:
        options = {"temperature": 0.2, "num_ctx": self.cfg.num_ctx}
        payload = {
            "model": self.cfg.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            # Constrained decoding against the pydantic schema — the model can
            # only emit JSON matching it, so parsing is not a hope.
            "format": schema.model_json_schema(),
            "options": options,
        }
        # Reasoning models (qwen3, deepseek-r1) want to emit a long <think> block
        # before answering. Under a schema constraint that thinking has nowhere to
        # go, and the model can grind for minutes producing very little. Ollama
        # exposes a switch; we only send it when asked, because older builds and
        # non-reasoning models reject an unknown value.
        if self.cfg.think is not None:
            payload["think"] = self.cfg.think

        req = urllib.request.Request(
            f"{self.cfg.ollama_host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout) as resp:
                body = json.loads(resp.read())
        except TimeoutError as exc:
            # socket.timeout is TimeoutError, NOT urllib.error.URLError, so this
            # used to escape as a raw traceback. Found on a first real review.
            raise OllamaError(
                f"{self.cfg.llm_model} did not answer within "
                f"{self.cfg.llm_timeout}s.\n"
                "  This is usually one of three things:\n"
                "    * a reasoning model fighting the JSON schema -- try "
                "`--model qwen2.5:7b-instruct`, or set SELFEVOLVE_LLM_THINK=false\n"
                "    * a cold model -- run `ollama run "
                f"{self.cfg.llm_model} \"\"` once, then retry\n"
                "    * a large file -- SELFEVOLVE_MAX_INPUT_CHARS caps how much "
                "is sent (currently "
                f"{self.cfg.max_input_chars})\n"
                "  Raise the ceiling with SELFEVOLVE_LLM_TIMEOUT if the model is "
                "simply slow."
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise OllamaError(
                f"could not reach Ollama at {self.cfg.ollama_host} ({exc}). "
                "Start it with `ollama serve`, or run with --backend fake."
            ) from exc

        if body.get("error"):
            raise OllamaError(f"Ollama returned an error: {body['error']}")
        content = body.get("message", {}).get("content", "")
        if not content.strip():
            raise OllamaError(
                f"{self.cfg.llm_model} returned an empty response. Reasoning "
                "models sometimes do this under a schema constraint -- try "
                "`--model qwen2.5:7b-instruct` or SELFEVOLVE_LLM_THINK=false."
            )
        return schema.model_validate_json(_strip_thinking(content))

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.cfg.ollama_host}/api/tags", timeout=2):
                return True
        except Exception:
            return False

    def models(self) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self.cfg.ollama_host}/api/tags", timeout=3) as resp:
                return [m["name"] for m in json.loads(resp.read()).get("models", [])]
        except Exception:
            return []


class FakeProvider:
    """Deterministic stand-in.

    It is not pretending to be smart. It exists so the loop is testable: it
    reads whatever lessons were injected into the prompt and demonstrably
    changes its output because of them, which is exactly the property the
    system is supposed to have.
    """

    name = "fake"

    def __init__(self, cfg: Config | None = None, scripted: list[dict[str, Any]] | None = None):
        self.cfg = cfg or DEFAULT
        self.scripted = list(scripted or [])
        self.calls: list[tuple[str, str]] = []

    def structured(self, system: str, prompt: str, schema: type[T]) -> T:
        self.calls.append((system, prompt))
        if self.scripted:
            return schema.model_validate(self.scripted.pop(0))
        return schema.model_validate(_fake_payload(schema, prompt))


def _fake_payload(schema: type[BaseModel], prompt: str) -> dict[str, Any]:
    fields = schema.model_fields
    if "items" in fields:
        # Honour any retrieved lesson that says "do not flag X".
        banned = _banned_topics(prompt)
        candidates = [
            {
                "title": "Unvalidated division",
                "body": "This divides without checking the denominator; a zero input raises.",
                "location": "line 3",
                "severity": "major",
                "tags": ["correctness"],
            },
            {
                "title": "Missing docstring",
                "body": "This function has no docstring.",
                "location": "line 1",
                "severity": "info",
                "tags": ["style", "docstring"],
            },
        ]
        kept = [
            c
            for c in candidates
            if not any(b in (c["title"] + " " + " ".join(c["tags"])).lower() for b in banned)
        ]
        return {"items": kept, "summary": f"{len(kept)} comment(s)."}
    if "rule" in fields:
        return _fake_rule(prompt)
    return {}


def _fake_rule(prompt: str) -> dict[str, str]:
    """Produce a rule that a later prompt can actually act on.

    A real model writes prose here. The fake one extracts the rejected comment's
    tags and emits a machine-followable "do not flag X" line, so `--backend fake`
    demonstrates the full loop — including a lesson visibly changing the next
    review — before you pull any model.
    """
    title = _field(prompt, "title:")
    tags = _field(prompt, "tags:")
    note = _field(prompt, "ENGINEER'S NOTE:")
    topic = (tags.split(",")[-1].strip() or title).lower() or "this class of comment"
    if "REJECTED" in prompt:
        return {
            "rule": f"Do not flag {topic} comments.",
            "rationale": note or "Rejected by the engineer as noise.",
            "scope_hint": "any",
        }
    if "REWROTE" in prompt:
        return {
            "rule": f"Keep raising {topic} concerns, but phrase them as the engineer corrected.",
            "rationale": note or "Concern was right, wording was not.",
            "scope_hint": "any",
        }
    return {
        "rule": f"Keep flagging {topic} issues; the engineer found this useful.",
        "rationale": note,
        "scope_hint": "any",
    }


def _field(prompt: str, label: str) -> str:
    for line in prompt.splitlines():
        if line.strip().startswith(label):
            return line.split(label, 1)[1].strip()
    return ""


def _banned_topics(prompt: str) -> list[str]:
    banned: list[str] = []
    for line in prompt.splitlines():
        low = line.lower()
        if "do not flag" in low or "don't flag" in low or "avoid flagging" in low:
            tail = low.split("flag", 1)[1]
            banned.extend(w.strip(" .,:;-") for w in tail.split() if len(w.strip(" .,:;-")) > 3)
    return banned


def _strip_thinking(content: str) -> str:
    """Reasoning models (qwen3, deepseek-r1) can emit <think> blocks before JSON."""
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1:
        content = content[start : end + 1]
    # validate early with a clearer error than pydantic's
    json.loads(content)
    return content


def get_provider(cfg: Config | None = None) -> LLMProvider:
    cfg = cfg or DEFAULT
    if cfg.llm_backend == "fake":
        return FakeProvider(cfg)
    return OllamaProvider(cfg)
