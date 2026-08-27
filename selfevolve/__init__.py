"""selfevolve — a self-evolving agent that runs entirely on your machine.

The loop (after ExpeL, Zhao et al. 2023): retrieve prior lessons -> act ->
collect a human verdict on each item -> reflect into reusable rules -> persist.
Model weights never change; the context the model receives gets better.

Offline by construction: one SQLite file for memory, local Ollama for generation
and embeddings, and no dependency that phones home. `harden_env()` runs on import
to close the telemetry paths our neighbours ship with; `selfevolve doctor` proves
the whole loop runs with all non-loopback sockets blocked.
"""

from .offline import harden_env

harden_env()  # before anything else imports a library that reads these vars

from .config import Config, DEFAULT  # noqa: E402
from .graph import SelfEvolvingAgent  # noqa: E402
from .models import Feedback, Insight, Item, Metrics, Scope, Trajectory  # noqa: E402
from .offline import NetworkBlocked, airgap  # noqa: E402
from .store import ExperienceStore  # noqa: E402
from .task import BaseTask, ItemsOut, RuleOut, Task, TaskInput  # noqa: E402

__version__ = "0.2.0"
__all__ = [
    "Config",
    "DEFAULT",
    "SelfEvolvingAgent",
    "ExperienceStore",
    "Feedback",
    "Insight",
    "Item",
    "Metrics",
    "Scope",
    "Trajectory",
    "BaseTask",
    "Task",
    "TaskInput",
    "ItemsOut",
    "RuleOut",
    "airgap",
    "NetworkBlocked",
    "harden_env",
]
