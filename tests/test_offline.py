"""Offline guarantees, enforced rather than asserted.

Every test here runs the real code with a socket guard installed that raises on
any connection to a non-loopback address. If a future edit adds a `requests.get`,
a telemetry ping, or a model download, these fail — on a developer's connected
laptop, in CI, anywhere. That is the point: "offline" claimed in a README rots,
"offline" enforced by a test does not.
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from selfevolve import Config, ExperienceStore, Feedback, NetworkBlocked, SelfEvolvingAgent, airgap
from selfevolve.embeddings import Embedder
from selfevolve.llm import FakeProvider, OllamaProvider
from selfevolve.offline import _ENV_LOCKDOWN, _STRIP_IF_PRESENT, report
from selfevolve.tasks.code_review import CodeReviewTask, input_from_file, input_from_text

CODE = "def ratio(a, b):\n    return a / b\n"
ROOT = Path(__file__).resolve().parents[1]


def _env(**overrides):
    """Inherit the real environment, then override.

    Passing a hand-built dict to subprocess REPLACES the environment entirely.
    On Windows that drops APPDATA, and Python locates per-user site-packages
    through APPDATA — so the child process could not find pydantic and died with
    ModuleNotFoundError. The test was broken, not the code. Inherit, then override.
    """
    import os

    env = dict(os.environ)
    env.setdefault("SELFEVOLVE_EMBED_BACKEND", "hash")
    env.setdefault("SELFEVOLVE_LLM_BACKEND", "fake")
    # Pin the child's stdout encoding. Without this it inherits the platform code
    # page -- cp1252 on a US Windows runner, UTF-8 on Linux -- and any test that
    # decodes child output has a different contract per platform. Tests that care
    # about a legacy code page override this explicitly.
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.update(overrides)
    return env



@pytest.fixture
def cfg(tmp_path):
    c = Config(data_dir=tmp_path / "store")
    c.llm_backend = "fake"
    c.embed_backend = "hash"
    return c


# --------------------------------------------------------------- the guard

def test_guard_blocks_external_and_allows_loopback():
    with airgap(allow_loopback=True):
        with pytest.raises(NetworkBlocked):
            socket.create_connection(("1.1.1.1", 80), timeout=1)
        with pytest.raises(NetworkBlocked):
            socket.socket().connect(("huggingface.co", 443))
        # loopback is permitted — Ollama lives there. Connection refused is fine;
        # what matters is that the guard didn't raise NetworkBlocked.
        try:
            socket.create_connection(("127.0.0.1", 1), timeout=0.2)
        except NetworkBlocked:
            pytest.fail("loopback was blocked but should be allowed")
        except OSError:
            pass


def test_guard_is_removed_on_exit():
    with airgap():
        pass
    with pytest.raises(OSError):  # a real failure, not NetworkBlocked
        socket.create_connection(("127.0.0.1", 1), timeout=0.2)


def test_strict_mode_blocks_even_loopback():
    with airgap(allow_loopback=False):
        with pytest.raises(NetworkBlocked):
            socket.create_connection(("127.0.0.1", 11434), timeout=1)


# ------------------------------------------------------- the loop, airgapped

def test_full_loop_runs_with_network_cut(cfg):
    """The headline claim: retrieve -> act -> verdict -> reflect -> persist,
    end to end, with every outbound socket blocked."""
    with airgap(allow_loopback=False):  # strictest: not even Ollama
        agent = SelfEvolvingAgent(CodeReviewTask(), cfg=cfg, llm=FakeProvider(cfg))
        ti = input_from_text(CODE, summary="ratio.py", project="p1")
        pending = agent.start(ti, thread_id="air1")
        assert pending["items"]

        agent.llm.scripted = [
            {
                "rule": "Do not flag missing docstring comments; the linter enforces them.",
                "rationale": "noise",
                "scope_hint": "any",
            }
        ]
        doc = next(i for i in pending["items"] if "docstring" in i.title.lower())
        result = agent.submit_feedback(
            [Feedback(item_id=doc.id, action="reject", note="linter covers it")], "air1"
        )
        assert result["learned"]

        # and the lesson is retrievable on a later question, still airgapped
        ti2 = input_from_text("def scale(x, y):\n    return x / y\n", project="p1")
        pending2 = agent.start(ti2, thread_id="air2")
        assert any("docstring" in i.rule.lower() for i in pending2["retrieved_insights"])
        assert not any("docstring" in i.title.lower() for i in pending2["items"])
        agent.close()


def test_store_and_embeddings_need_no_network(cfg):
    with airgap(allow_loopback=False):
        store = ExperienceStore(cfg)
        emb = Embedder(cfg)
        assert emb.backend == "hash"
        vec = emb.embed_one("pyspark broadcast join")
        assert len(vec) == 384
        assert abs(sum(v * v for v in vec) - 1.0) < 1e-6  # normalized
        store.close()


def test_embedding_cache_survives_and_avoids_recompute(cfg):
    store = ExperienceStore(cfg)
    with airgap(allow_loopback=False):
        v1 = store.embedder.embed_one("a rule about broadcast joins")
        v2 = store.embedder.embed_one("a rule about broadcast joins")
        assert v1 == v2
    rows = store.conn.execute("SELECT COUNT(*) c FROM embed_cache").fetchone()["c"]
    assert rows >= 1, "embedding was not cached"
    store.close()


def test_ollama_provider_fails_cleanly_when_unreachable(cfg):
    """Offline must degrade with a useful message, not a stack trace."""
    from selfevolve.llm import OllamaError
    from selfevolve.task import ItemsOut

    cfg.ollama_host = "http://127.0.0.1:1"
    with airgap(allow_loopback=True):
        with pytest.raises(OllamaError, match="could not reach Ollama"):
            OllamaProvider(cfg).structured("s", "p", ItemsOut)


def test_embedder_falls_back_when_ollama_dies_midway(cfg):
    """A lesson must never be lost because the embedder went away."""
    cfg.embed_backend = "ollama"
    cfg.ollama_host = "http://127.0.0.1:1"
    with airgap(allow_loopback=True):
        emb = Embedder(cfg)
        vec = emb.embed_one("some rule text")
        assert len(vec) == 384
        assert emb.backend == "hash", "should have flipped to the fallback"


# ------------------------------------------------------------ configuration

def test_env_is_hardened_on_import():
    import os

    for key in _ENV_LOCKDOWN:
        assert os.environ.get(key) is not None, f"{key} was not set"
    for key in _STRIP_IF_PRESENT:
        assert not os.environ.get(key), f"{key} should have been stripped"


def test_all_configured_hosts_are_local():
    info = report()
    assert info["ollama_is_local"] is True
    assert info["leaked_keys"] == []


def test_no_remote_urls_in_source():
    """Grep our own source for anything that looks like an outbound endpoint.

    Comments and docstrings may reference URLs (a paper, a repo); executable
    lines may not. Crude, and that's fine — it catches the realistic mistake,
    which is someone pasting in an API call.
    """
    offenders = []
    for path in (ROOT / "selfevolve").rglob("*.py"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                continue
            # A scheme literal with no host after it ("http://") is a prefix
            # check, not an endpoint. Only flag a scheme followed by a hostname.
            for m in re.finditer(r"https?://([A-Za-z0-9._-]+)", line):
                host = m.group(1)
                if host in ("127.0.0.1", "localhost"):
                    continue
                offenders.append(f"{path.name}:{n}: {stripped[:90]}")
    assert not offenders, "remote URLs in executable code:\n" + "\n".join(offenders)


BANNED = {
    "requests", "httpx", "aiohttp", "urllib3", "huggingface_hub",
    "sentence_transformers", "chromadb", "langchain", "langsmith", "langgraph",
    "openai", "anthropic", "posthog", "onnxruntime",
}


def test_our_source_imports_nothing_that_talks_to_the_network():
    """Parse every module we ship and check its import statements.

    This replaced a check on sys.modules after importing selfevolve. That version
    was unreliable in a shared environment: on a machine with a hundred packages
    in user site-packages, something else can drag `requests` in through a .pth
    file or a plugin entry point, and the test then fails for a dependency that
    is not ours. A test that cries wolf is a test people learn to ignore.

    An AST scan asks the question we actually mean -- does OUR code import this --
    and the answer does not depend on what else is installed.
    """
    import ast

    offenders = []
    for path in (ROOT / "selfevolve").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in BANNED:
                    offenders.append(f"{path.name}:{node.lineno}: import {name}")
    assert not offenders, "network libraries imported by our code:\n" + "\n".join(offenders)


def test_declared_dependencies_pull_nothing_banned():
    """The installed tree matters as much as our own imports: a dependency that
    drags in an HTTP client puts one on the machine whether we call it or not.

    Checks selfevolve's own declared requirements. The airgap workflow proves the
    stronger version -- a fresh venv install containing none of these -- because
    only a clean environment can answer the transitive question honestly.
    """
    from importlib.metadata import requires

    declared = requires("selfevolve") or []
    runtime = [
        r.split(";")[0].strip() for r in declared
        if "extra ==" not in r  # optional extras (streamlit, pytest) are not runtime
    ]
    offenders = [r for r in runtime if r.split()[0].split("[")[0].lower() in BANNED]
    assert not offenders, f"declared runtime dependencies include: {offenders}"


def test_doctor_exits_zero(tmp_path):
    env = _env(SELFEVOLVE_DATA_DIR=str(tmp_path / "doc"))
    out = subprocess.run(
        [sys.executable, "-m", "selfevolve.cli", "doctor"],
        capture_output=True, cwd=ROOT, env=env, encoding="utf-8",
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Full loop completed with the network cut" in out.stdout


def test_doctor_leaves_no_rules_behind(tmp_path):
    """A diagnostic that pollutes your learned rules is a bad diagnostic."""
    data_dir = tmp_path / "doc2"
    env = _env(SELFEVOLVE_DATA_DIR=str(data_dir))
    subprocess.run(
        [sys.executable, "-m", "selfevolve.cli", "doctor"],
        capture_output=True, cwd=ROOT, env=env, check=True, encoding="utf-8",
    )
    cfg = Config(data_dir=data_dir)
    store = ExperienceStore(cfg)
    assert store.all_insights(include_retired=True) == []
    assert store.all_trajectories() == []
    store.close()


# ------------------------------------------------------------ data ownership

def test_everything_lives_in_one_portable_file(cfg):
    with airgap(allow_loopback=False):
        agent = SelfEvolvingAgent(CodeReviewTask(), cfg=cfg, llm=FakeProvider(cfg))
        pending = agent.start(input_from_text(CODE, project="p1"), "port1")
        agent.submit_feedback(
            [Feedback(item_id=pending["items"][0].id, action="reject", note="x")], "port1"
        )
        agent.close()

    db = cfg.db_path()
    assert db.exists() and db.stat().st_size > 0

    # copy the file elsewhere and everything is still there — the whole backup story
    copy = cfg.data_dir.parent / "copied.db"
    copy.write_bytes(db.read_bytes())
    moved = Config(data_dir=copy.parent)
    moved.db_file = copy.name
    moved.embed_backend = "hash"
    reopened = ExperienceStore(moved)
    assert len(reopened.all_insights()) >= 1
    assert len(reopened.all_trajectories()) == 1
    reopened.close()


# --------------------------------------------------- platform / environment

def test_cloud_sync_folder_is_detected(tmp_path):
    """SQLite inside OneDrive/Dropbox fails intermittently and blames us for it.
    Warn instead of letting someone debug 'database is locked' for an evening."""
    for path, expected in [
        (r"C:/Users/y/OneDrive/Desktop/proj/.selfevolve", "OneDrive"),
        ("/home/y/Dropbox/proj/.selfevolve", "Dropbox"),
        ("/Users/y/Library/Mobile Documents/com~apple~CloudDocs/se", "iCloud Drive"),
    ]:
        warning = Config(data_dir=Path(path)).sync_folder_warning()
        assert warning and expected in warning, f"missed {expected} in {path}"
    assert Config(data_dir=tmp_path / "plain").sync_folder_warning() is None


def test_output_has_no_escape_codes_when_piped():
    """Piping a command must produce a clean file, not escape-code soup —
    the default on a Windows console that hasn't enabled VT."""
    env = _env(SELFEVOLVE_DATA_DIR=str(Path(tempfile.mkdtemp()) / "pipe"))
    out = subprocess.run(
        [sys.executable, "-m", "selfevolve.cli", "metrics"],
        capture_output=True, cwd=ROOT, env=env, check=True, encoding="utf-8",
    )
    assert "\033[" not in out.stdout, "ANSI codes leaked into piped output"


def test_published_modelfile_matches_the_task():
    """A published Ollama model that hard-codes a stale copy of the system prompt
    is worse than none — it claims to come from this repo while behaving
    differently. Fail the build rather than let the two drift."""
    out = subprocess.run(
        [sys.executable, "scripts/build_modelfile.py", "--check"],
        capture_output=True, cwd=ROOT, encoding="utf-8",
    )
    assert out.returncode == 0, out.stdout + out.stderr


def test_doctor_survives_a_legacy_console_encoding():
    """Windows pipes stdout as cp1252, which cannot encode U+2713.

    `doctor` printed a check mark, raised UnicodeEncodeError, and then crashed
    again inside its own error handler -- which also printed a glyph -- so the
    real failure was never shown. Found on a real Windows 11 machine; this pins
    it shut.
    """
    env = _env(
        SELFEVOLVE_DATA_DIR=str(Path(tempfile.mkdtemp()) / "cp1252"),
        PYTHONIOENCODING="cp1252",
    )
    # Decode as cp1252 too: the child is writing cp1252 bytes, so decoding them
    # as UTF-8 would fail in the test harness rather than in the code under test.
    out = subprocess.run(
        [sys.executable, "-m", "selfevolve.cli", "doctor"],
        capture_output=True, cwd=ROOT, env=env,
        encoding="cp1252", errors="replace",
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "UnicodeEncodeError" not in out.stderr
    assert "Full loop completed with the network cut" in out.stdout


def test_timeout_is_reported_as_guidance_not_a_traceback(cfg):
    """A slow local model must produce advice, not a stack trace.

    socket.timeout is TimeoutError, which is NOT urllib.error.URLError, so the
    original handler missed it entirely and a first real review ended in a raw
    traceback from http.client. Found on a real machine running qwen3:8b.
    """
    from unittest.mock import patch

    from selfevolve.llm import OllamaError, OllamaProvider
    from selfevolve.task import ItemsOut

    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        with pytest.raises(OllamaError) as err:
            OllamaProvider(cfg).structured("s", "p", ItemsOut)

    msg = str(err.value)
    assert "did not answer within" in msg
    assert "qwen2.5:7b-instruct" in msg, "should suggest a concrete alternative"
    assert "SELFEVOLVE_LLM_TIMEOUT" in msg, "should name the knob that raises the ceiling"


def test_empty_model_response_is_explained(cfg):
    """Reasoning models sometimes return nothing under a schema constraint.
    Pydantic's 'Invalid JSON' is not a useful thing to show for that."""
    import io
    from unittest.mock import patch

    from selfevolve.llm import OllamaError, OllamaProvider
    from selfevolve.task import ItemsOut

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    body = b'{"message": {"content": "   "}}'
    with patch("urllib.request.urlopen", return_value=_Resp(body)):
        with pytest.raises(OllamaError, match="empty response"):
            OllamaProvider(cfg).structured("s", "p", ItemsOut)


def test_oversized_file_is_capped_and_reported(tmp_path):
    """A 2,000-line module should be trimmed, and the caller told it was."""
    big = tmp_path / "big.py"
    big.write_text("\n".join(f"x{i} = {i}" for i in range(4000)), encoding="utf-8")

    ti = input_from_file(big, project="p", max_chars=5000)
    assert len(ti.text) <= 5000
    assert ti.meta["truncated_chars"] > 0
    assert ti.meta["total_lines"] == 4000
    assert not ti.text.endswith("|"), "should cut on a line boundary"

    untouched = input_from_file(big, project="p", max_chars=None)
    assert untouched.meta["truncated_chars"] == 0


def test_model_card_names_the_model_we_actually_ship():
    """A model card is marketing copy for a real artifact. If it names a model
    the Modelfile does not use, it is describing something that doesn't exist."""
    from selfevolve.config import Config

    card = (ROOT / "ollama" / "MODEL_CARD.md").read_text(encoding="utf-8")
    modelfile = (ROOT / "ollama" / "Modelfile").read_text(encoding="utf-8")
    model = Config().llm_model

    assert f"FROM {model}" in modelfile
    assert model in card, f"MODEL_CARD.md does not mention {model}"
    assert "does not learn" in card, "the card must be explicit about what it is not"


def test_ollama_readme_has_no_stale_model_reference():
    readme = (ROOT / "ollama" / "README.md").read_text(encoding="utf-8")
    assert "qwen3:8b" not in readme, "stale base model in ollama/README.md"
