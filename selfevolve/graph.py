"""The ExpeL loop as an explicit, checkpointed state machine.

    retrieve -> act -> [PAUSE for human verdict] -> reflect -> persist

This was written on LangGraph first. It isn't any more, and the reason is worth
stating plainly: `langgraph` pulls in `langchain-core`, which pulls in
`langsmith`, which pulls in `requests` and `httpx` — a tracing client that ships
data to a vendor, sitting in the import graph of a system whose entire premise is
that it runs on your machine. Disabling it with an environment variable is a
runtime flag, not a guarantee. Not shipping it is a guarantee.

What LangGraph was actually providing here was `interrupt()` and a checkpointer.
For a five-node linear graph, both are a table and about eighty lines. The
dependency tree is now: pydantic. That's it.

Two ordering decisions survive unchanged, because they're the design:

1. `retrieve` runs before `act`, unconditionally. It is not a tool the model may
   choose to call. Every episode checks for relevant experience, so behaviour
   never depends on the model remembering to look things up.

2. `reflect` runs after the human verdict, never before. The agent cannot certify
   its own output as correct — a person supplies that judgement.

Between them the loop genuinely stops. `start()` writes the episode to SQLite and
returns; the process can exit. `submit_feedback()` with the same thread_id — a
minute or a week later, from the CLI or Streamlit — resumes exactly where it left
off.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .config import DEFAULT, Config
from .llm import LLMProvider, get_provider
from .models import Feedback, Insight, Item, Scope, Trajectory
from .store import ExperienceStore
from .task import BaseTask, Task, TaskInput

EPISODE_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    thread_id   TEXT PRIMARY KEY,
    task        TEXT NOT NULL,
    status      TEXT NOT NULL,          -- awaiting_feedback | complete
    state       TEXT NOT NULL,          -- JSON snapshot of the paused loop
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status, updated_at DESC);
"""


class SelfEvolvingAgent:
    def __init__(
        self,
        task: Task,
        cfg: Config | None = None,
        store: ExperienceStore | None = None,
        llm: LLMProvider | None = None,
    ):
        self.cfg = cfg or DEFAULT
        self.task = task
        self.store = store or ExperienceStore(self.cfg)
        self.llm = llm or get_provider(self.cfg)
        # Paused episodes live in the SAME file as the learned rules, so the
        # whole system's state is one path on disk. Copy experience.db and you
        # have carried the agent's memory and its in-flight reviews together.
        self._ck = self.store.conn
        self._ck.executescript(EPISODE_SCHEMA)
        self._ck.commit()

    # ------------------------------------------------------------------ nodes

    def retrieve_node(self, ti: TaskInput) -> tuple[list[Insight], list[Trajectory]]:
        query = self.task.retrieval_text(ti)
        return (
            self.store.retrieve_insights(query, scope=ti.scope),
            self.store.retrieve_trajectories(query, scope=ti.scope),
        )

    def act_node(
        self, ti: TaskInput, insights: list[Insight], trajectories: list[Trajectory]
    ) -> tuple[list[Item], str]:
        schema = self.task.output_schema()
        prompt = self.task.act_prompt(ti, insights, trajectories)
        out = self.llm.structured(self.task.system_prompt(), prompt, schema)
        return self.task.to_items(out), getattr(out, "summary", "")

    def reflect_node(
        self, ti: TaskInput, items: dict[str, Item], feedback: list[Feedback]
    ) -> list[Insight]:
        schema = self.task.reflect_schema()
        learned: list[Insight] = []
        for fb in feedback:
            item = items.get(fb.item_id)
            if item is None:
                continue
            if fb.action == "accept" and not fb.note:
                # An unexplained accept confirms existing behaviour. Minting a
                # rule from it is how you end up with a hundred restatements of
                # "flagging real bugs is good".
                continue
            prompt = self.task.reflect_prompt(ti, item, fb)
            try:
                out = self.llm.structured(_REFLECT_SYSTEM, prompt, schema)
            except Exception as exc:  # a bad reflection must not lose the episode
                learned.append(
                    Insight(
                        rule=f"[unparsed reflection] {fb.action} on '{item.title}': {fb.note}"[:400],
                        rationale=f"reflection failed: {exc}"[:300],
                        origin_action=fb.action,
                        scope=ti.scope,
                        confidence=0.3,
                    )
                )
                continue
            ins = self.task.to_insight(out, fb, ti, item)
            if ins.rule:
                learned.append(ins)
        return learned

    def persist_node(
        self,
        ti: TaskInput,
        items: list[Item],
        feedback: list[Feedback],
        learned: list[Insight],
    ) -> Trajectory:
        stored_ids = [self.store.add_insight(ins).id for ins in learned]
        traj = Trajectory(
            task=self.task.name,
            scope=ti.scope,
            input_summary=ti.summary or ti.text[:120],
            input_text=self.task.retrieval_text(ti),
            items=items,
            feedback=feedback,
            insight_ids=stored_ids,
        )
        self.store.add_trajectory(traj)
        return traj

    # ------------------------------------------------------------------- api

    def start(self, task_input: TaskInput, thread_id: str) -> dict[str, Any]:
        """Run retrieve + act, checkpoint, and pause for a human verdict."""
        insights, trajectories = self.retrieve_node(task_input)
        items, summary = self.act_node(task_input, insights, trajectories)

        self._save(
            thread_id,
            "awaiting_feedback",
            {
                "task_input": task_input.model_dump(),
                "items": [i.model_dump() for i in items],
                "summary": summary,
                "retrieved_insights": [i.model_dump() for i in insights],
            },
        )
        return {
            "items": items,
            "summary": summary,
            "retrieved_insights": insights,
            "awaiting_feedback": True,
            "thread_id": thread_id,
        }

    def submit_feedback(self, feedback: list[Feedback], thread_id: str) -> dict[str, Any]:
        """Resume a paused episode with human verdicts; runs reflect + persist."""
        state = self._load(thread_id)
        if state is None:
            raise KeyError(f"no paused episode for thread {thread_id!r}")

        ti = TaskInput(**state["task_input"])
        items = [Item(**i) for i in state["items"]]
        by_id = {i.id: i for i in items}

        # Verdicts on items that aren't in this episode are dropped, and items
        # nobody ruled on are simply absent — never silently counted as accepted.
        # An unreviewed comment teaches nothing, and pretending otherwise is how
        # these systems learn noise.
        feedback = [f for f in feedback if f.item_id in by_id]
        counts = {"accept": 0, "reject": 0, "edit": 0}
        for f in feedback:
            counts[f.action] += 1

        learned = self.reflect_node(ti, by_id, feedback)
        traj = self.persist_node(ti, items, feedback, learned)

        state.update(
            {
                "feedback": [f.model_dump() for f in feedback],
                "counts": counts,
                "trajectory_id": traj.id,
                "learned": [i.model_dump() for i in learned],
            }
        )
        self._save(thread_id, "complete", state)

        return {
            "trajectory_id": traj.id,
            "counts": counts,
            "learned": learned,
            "items": items,
            "thread_id": thread_id,
        }

    def pending(self, thread_id: str) -> dict[str, Any] | None:
        state = self._load(thread_id)
        if state is None:
            return None
        return {
            "items": [Item(**i) for i in state["items"]],
            "summary": state.get("summary", ""),
            "retrieved_insights": [Insight(**i) for i in state.get("retrieved_insights", [])],
            "awaiting_feedback": True,
            "thread_id": thread_id,
        }

    def open_episodes(self) -> list[dict[str, Any]]:
        """Every review you started and never ruled on. Without this they're
        invisible — a paused episode is easy to forget it exists."""
        rows = self._ck.execute(
            "SELECT thread_id, task, created_at, state FROM episodes"
            " WHERE status='awaiting_feedback' ORDER BY updated_at DESC"
        ).fetchall()
        out = []
        for r in rows:
            state = json.loads(r["state"])
            out.append(
                {
                    "thread_id": r["thread_id"],
                    "task": r["task"],
                    "created_at": r["created_at"],
                    "summary": state.get("task_input", {}).get("summary", ""),
                    "n_items": len(state.get("items", [])),
                }
            )
        return out

    def close(self) -> None:
        """The agent shares the store's connection, so closing the agent closes
        the store. Callers that still need the store should close that instead."""
        self.store.close()

    # -------------------------------------------------------- checkpointing

    def _save(self, thread_id: str, status: str, state: dict[str, Any]) -> None:
        now = time.time()
        self._ck.execute(
            "INSERT INTO episodes(thread_id, task, status, state, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?)"
            " ON CONFLICT(thread_id) DO UPDATE SET status=excluded.status,"
            " state=excluded.state, updated_at=excluded.updated_at",
            (thread_id, self.task.name, status, json.dumps(state), now, now),
        )
        self._ck.commit()

    def _load(self, thread_id: str) -> dict[str, Any] | None:
        row = self._ck.execute(
            "SELECT state FROM episodes WHERE thread_id=? AND status='awaiting_feedback'",
            (thread_id,),
        ).fetchone()
        return json.loads(row["state"]) if row else None


_REFLECT_SYSTEM = (
    "You turn a single piece of engineer feedback into one short, reusable rule "
    "for a future review. Write the rule as an instruction, not as a description "
    "of what happened. It must be specific enough to change behaviour and general "
    "enough to apply to different code with different names. Never mention this "
    "particular file, function, or variable."
)


__all__ = ["SelfEvolvingAgent", "BaseTask", "Task", "TaskInput", "Scope"]
