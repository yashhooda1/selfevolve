"""Force the deterministic, service-free backends for every test.

Set here rather than in the shell so `pytest` behaves identically on a machine
with Ollama running and one without — a test suite whose results depend on which
daemons happen to be up is worse than no test suite.
"""

import os

os.environ.setdefault("SELFEVOLVE_EMBED_BACKEND", "hash")
os.environ.setdefault("SELFEVOLVE_LLM_BACKEND", "fake")
