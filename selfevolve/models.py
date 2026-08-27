"""Core data models for the self-evolving agent loop.

Everything the loop passes around is a pydantic model so that the same schemas
can be used for (a) validating LLM structured output, (b) persisting to the
vector store, and (c) rendering in the Streamlit UI.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

Action = Literal["accept", "reject", "edit"]


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Scope(BaseModel):
    """Where a lesson applies.

    A rule learned on one repo/language/framework is not automatically valid
    everywhere. Scope is stored on every insight and used to filter retrieval,
    which is the single biggest difference between a toy ExpeL loop and one you
    can actually leave running on a team.
    """

    project: str = "default"
    language: str = "any"
    framework: str = "any"
    team: str = "any"

    def as_metadata(self) -> dict[str, str]:
        return {
            "scope_project": self.project,
            "scope_language": self.language,
            "scope_framework": self.framework,
            "scope_team": self.team,
        }

    @classmethod
    def from_metadata(cls, md: dict[str, Any]) -> Scope:
        return cls(
            project=md.get("scope_project", "default"),
            language=md.get("scope_language", "any"),
            framework=md.get("scope_framework", "any"),
            team=md.get("scope_team", "any"),
        )

    def key(self) -> str:
        return f"{self.project}/{self.language}/{self.framework}/{self.team}"


class Item(BaseModel):
    """One reviewable unit of agent output.

    In the code-review task this is a review comment. In a writing task it would
    be a suggested edit; in a research task, a claim. The loop only requires that
    output can be decomposed into items a human can accept / reject / edit.
    """

    id: str = Field(default_factory=lambda: _new_id("item"))
    title: str = ""
    body: str = ""
    location: str = ""
    severity: Literal["info", "minor", "major", "critical"] = "minor"
    tags: list[str] = Field(default_factory=list)

    def render(self) -> str:
        loc = f" ({self.location})" if self.location else ""
        return f"[{self.severity}] {self.title}{loc}\n{self.body}"


class Feedback(BaseModel):
    """A human's verdict on one item.

    `note` is the highest-value field in the whole system. A rejection tells the
    agent a comment failed; a rejection with a reason tells it *why*, which is
    what makes the resulting rule reusable instead of a blanket suppression.
    """

    item_id: str
    action: Action
    edited_body: str = ""
    note: str = ""


class Insight(BaseModel):
    """A short, reusable rule distilled from one item + its feedback."""

    id: str = Field(default_factory=lambda: _new_id("ins"))
    rule: str
    rationale: str = ""
    origin_action: Action = "accept"
    scope: Scope = Field(default_factory=Scope)
    confidence: float = 0.5
    support: int = 1
    contradictions: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    retired: bool = False

    def fingerprint(self) -> str:
        """Stable hash of (normalized rule, scope) used to merge duplicates."""
        norm = " ".join(self.rule.lower().split())
        return hashlib.sha1(f"{norm}|{self.scope.key()}".encode()).hexdigest()[:16]

    def as_metadata(self) -> dict[str, Any]:
        md: dict[str, Any] = {
            "kind": "insight",
            "rule": self.rule,
            "rationale": self.rationale,
            "origin_action": self.origin_action,
            "confidence": float(self.confidence),
            "support": int(self.support),
            "contradictions": int(self.contradictions),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "retired": bool(self.retired),
            "fingerprint": self.fingerprint(),
        }
        md.update(self.scope.as_metadata())
        return md

    @classmethod
    def from_record(cls, rec_id: str, md: dict[str, Any]) -> Insight:
        return cls(
            id=rec_id,
            rule=md.get("rule", ""),
            rationale=md.get("rationale", ""),
            origin_action=md.get("origin_action", "accept"),
            scope=Scope.from_metadata(md),
            confidence=float(md.get("confidence", 0.5)),
            support=int(md.get("support", 1)),
            contradictions=int(md.get("contradictions", 0)),
            created_at=float(md.get("created_at", 0.0)),
            updated_at=float(md.get("updated_at", 0.0)),
            retired=bool(md.get("retired", False)),
        )


class Trajectory(BaseModel):
    """A full episode: what came in, what the agent said, what the human did."""

    id: str = Field(default_factory=lambda: _new_id("traj"))
    task: str
    scope: Scope = Field(default_factory=Scope)
    input_summary: str = ""
    input_text: str = ""
    items: list[Item] = Field(default_factory=list)
    feedback: list[Feedback] = Field(default_factory=list)
    insight_ids: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)

    def counts(self) -> dict[str, int]:
        out = {"accept": 0, "reject": 0, "edit": 0}
        for fb in self.feedback:
            out[fb.action] = out.get(fb.action, 0) + 1
        return out

    def render(self, max_items: int = 4) -> str:
        by_id = {i.id: i for i in self.items}
        lines = [f"Past review of: {self.input_summary}"]
        for fb in self.feedback[:max_items]:
            item = by_id.get(fb.item_id)
            if not item:
                continue
            verdict = fb.action.upper()
            lines.append(f"- {verdict}: {item.title} — {item.body[:180]}")
            if fb.note:
                lines.append(f"  engineer said: {fb.note}")
            if fb.action == "edit" and fb.edited_body:
                lines.append(f"  corrected to: {fb.edited_body[:180]}")
        return "\n".join(lines)


class Metrics(BaseModel):
    reviews: int = 0
    items: int = 0
    accepted: int = 0
    rejected: int = 0
    edited: int = 0
    insights: int = 0

    @property
    def acceptance_rate(self) -> float:
        total = self.accepted + self.rejected + self.edited
        return (self.accepted / total) if total else 0.0

    @property
    def false_positive_rate(self) -> float:
        total = self.accepted + self.rejected + self.edited
        return (self.rejected / total) if total else 0.0
