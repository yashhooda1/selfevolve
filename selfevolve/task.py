"""The seam that makes this a framework rather than one code-review agent.

A Task answers five questions:

  1. what is the unit of work?              -> TaskInput
  2. how do we index it for retrieval?      -> retrieval_text / scope
  3. what does the agent produce?           -> output_schema + act_prompt
  4. how does that decompose into items a
     human can accept/reject/edit?          -> to_items
  5. how does one item + one verdict become
     a reusable rule?                       -> reflect_prompt

Everything else — retrieval, the human-in-the-loop pause, merging, decay,
metrics — is domain-independent and lives in the graph.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .models import Feedback, Insight, Item, Scope, Trajectory


class TaskInput(BaseModel):
    """Whatever the agent is asked to work on this round."""

    text: str
    summary: str = ""
    scope: Scope = Field(default_factory=Scope)
    meta: dict[str, Any] = Field(default_factory=dict)


class ItemsOut(BaseModel):
    """Default structured-output schema: a list of reviewable items."""

    items: list[Item] = Field(default_factory=list)
    summary: str = ""


class RuleOut(BaseModel):
    """Default reflection schema: one distilled rule."""

    rule: str
    rationale: str = ""
    scope_hint: str = "any"


@runtime_checkable
class Task(Protocol):
    name: str

    def system_prompt(self) -> str: ...

    def retrieval_text(self, task_input: TaskInput) -> str: ...

    def act_prompt(
        self, task_input: TaskInput, insights: list[Insight], trajectories: list[Trajectory]
    ) -> str: ...

    def output_schema(self) -> type[BaseModel]: ...

    def to_items(self, output: BaseModel) -> list[Item]: ...

    def reflect_prompt(self, task_input: TaskInput, item: Item, feedback: Feedback) -> str: ...

    def reflect_schema(self) -> type[BaseModel]: ...

    def to_insight(self, output: BaseModel, feedback: Feedback, task_input: TaskInput) -> Insight: ...


class BaseTask:
    """Sensible defaults for the parts most tasks don't need to customise."""

    name = "base"

    def output_schema(self) -> type[BaseModel]:
        return ItemsOut

    def reflect_schema(self) -> type[BaseModel]:
        return RuleOut

    def to_items(self, output: BaseModel) -> list[Item]:
        return list(getattr(output, "items", []))

    def retrieval_text(self, task_input: TaskInput) -> str:
        return task_input.text[:4000]

    def to_insight(self, output: BaseModel, feedback: Feedback, task_input: TaskInput) -> Insight:
        hint = (getattr(output, "scope_hint", "any") or "any").strip().lower()
        scope = task_input.scope.model_copy(deep=True)
        if hint in ("any", "global", "all"):
            # A lesson the engineer framed generally shouldn't be locked to one repo.
            scope = Scope(
                project="any", language=scope.language, framework=scope.framework, team=scope.team
            )
        confidence = {"accept": 0.6, "edit": 0.55, "reject": 0.5}[feedback.action]
        return Insight(
            rule=getattr(output, "rule", "").strip(),
            rationale=getattr(output, "rationale", "").strip(),
            origin_action=feedback.action,
            scope=scope,
            confidence=confidence,
        )

    @staticmethod
    def render_lessons(insights: list[Insight], trajectories: list[Trajectory]) -> str:
        parts: list[str] = []
        if insights:
            parts.append("LESSONS FROM PAST REVIEWS (follow these):")
            for i, ins in enumerate(insights, 1):
                marker = {"accept": "reinforce", "reject": "avoid", "edit": "correct"}[
                    ins.origin_action
                ]
                parts.append(
                    f"{i}. [{marker}, confidence {ins.confidence:.2f}, seen {ins.support}x] {ins.rule}"
                    + (f" — {ins.rationale}" if ins.rationale else "")
                )
        if trajectories:
            parts.append("\nSIMILAR PAST REVIEWS AND HOW THE ENGINEER RESPONDED:")
            for traj in trajectories:
                parts.append(traj.render())
        if not parts:
            parts.append("No prior experience is available yet. Review from first principles.")
        return "\n".join(parts)
