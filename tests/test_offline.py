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
from pathlib import Path

import pytest

from selfevolve import Config, ExperienceStore, Feedback, NetworkBlocked, SelfEvolvingAgent, airgap
from selfevolve.embeddings import Embedder
from selfevolve.llm import FakeProvider, OllamaProvider
from selfevolve.offline import _ENV_LOCKDOWN, _STRIP_IF_PRESENT, report
from selfevolve.tasks.code_review import CodeReviewTask, input_from_text

CODE = "def ratio(a, b):\n    return a / b\n"
ROOT = Path(__file__).resolve().parents[1]


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

    for key, expected in _ENV_LOCKDOWN.items():
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
        for n, line in enumerate(path.read_text().splitlines(), 1):
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


def test_no_network_libraries_imported():
    """requests / httpx / huggingface_hub must not be in the import graph."""
    banned = {"requests", "httpx", "aiohttp", "huggingface_hub", "sentence_transformers", "chromadb"}
    code = (
        "import selfevolve, selfevolve.store, selfevolve.graph, selfevolve.cli, "
        "selfevolve.tasks.code_review, sys; "
        f"print(','.join(sorted({banned!r} & set(sys.modules))))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True
    )
    assert out.stdout.strip() == "", f"network libraries imported: {out.stdout.strip()}"


def test_doctor_exits_zero(tmp_path):
    env = {
        "SELFEVOLVE_DATA_DIR": str(tmp_path / "doc"),
        "SELFEVOLVE_EMBED_BACKEND": "hash",
        "SELFEVOLVE_LLM_BACKEND": "fake",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp_path),
    }
    out = subprocess.run(
        [sys.executable, "-m", "selfevolve.cli", "doctor"],
        capture_output=True, text=True, cwd=ROOT, env=env,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Full loop completed with the network cut" in out.stdout


def test_doctor_leaves_no_rules_behind(tmp_path):
    """A diagnostic that pollutes your learned rules is a bad diagnostic."""
    data_dir = tmp_path / "doc2"
    env = {
        "SELFEVOLVE_DATA_DIR": str(data_dir),
        "SELFEVOLVE_EMBED_BACKEND": "hash",
        "SELFEVOLVE_LLM_BACKEND": "fake",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp_path),
    }
    subprocess.run(
        [sys.executable, "-m", "selfevolve.cli", "doctor"],
        capture_output=True, text=True, cwd=ROOT, env=env, check=True,
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
