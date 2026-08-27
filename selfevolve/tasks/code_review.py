"""Reference Task: a code-review agent that learns a team's review standards.

The model supplies general code knowledge. The feedback supplies how *this* team
wants that knowledge applied — which warnings they care about, which they
consider noise, and how they want a recommendation phrased.
"""

from __future__ import annotations

from pathlib import Path

from ..models import Feedback, Item, Scope
from ..task import BaseTask, TaskInput

_EXT_LANG = {
    ".py": "python",
    ".sql": "sql",
    ".scala": "scala",
    ".java": "java",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".sh": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
}


class CodeReviewTask(BaseTask):
    """Two modes, because a review agent has two jobs at different times.

    The default prompt is tuned hard against false positives -- explicitly told
    not to invent problems, and that returning nothing is a valid answer. That is
    right for steady-state use: once the agent knows your standards, silence is
    informative.

    It is wrong for bootstrapping. On a small local model that pressure tips into
    saying nothing at all, and an agent that never comments can never learn --
    there is nothing to rule on, so no lesson is ever created and the loop is
    starved. `thorough=True` asks for candidates even when minor, accepting more
    noise in exchange for something to give verdicts on. Use it to seed an empty
    store, then drop back to the default.
    """

    name = "code_review"

    def __init__(self, thorough: bool = False):
        self.thorough = thorough

    def system_prompt(self) -> str:
        if self.thorough:
            return (
                "You are a senior code reviewer doing a first pass on an unfamiliar "
                "codebase. Surface every concern worth a moment of an engineer's "
                "attention: correctness, error handling, resource leaks, silent "
                "failures, unvalidated input, performance traps, unclear naming. "
                "Include minor ones — a human will decide what matters and their "
                "verdicts are what the system learns from. Prefer 3-6 comments over "
                "none. Still do not restate what the code does, and do not flag "
                "formatting a linter already handles. Lessons from past reviews on "
                "this codebase override your defaults."
            )
        return (
            "You are a senior code reviewer. You produce a small number of "
            "high-signal comments. Each comment states a concrete concern and a "
            "specific recommendation. You do not comment on style that a "
            "formatter or linter already enforces, you do not restate what the "
            "code does, and you do not invent problems to fill space. If the "
            "code is fine, return no comments. Lessons from past reviews on this "
            "codebase override your defaults."
        )

    def retrieval_text(self, task_input: TaskInput) -> str:
        # Index on the code itself plus its language, so a lesson learned on one
        # function is retrievable for a different function that rhymes with it.
        head = f"{task_input.scope.language} {task_input.scope.framework}\n"
        return (head + task_input.text)[:4000]

    def act_prompt(self, task_input: TaskInput, insights, trajectories) -> str:
        return f"""{self.render_lessons(insights, trajectories)}

---
Review the following {task_input.scope.language} code from project "{task_input.scope.project}".

```
{task_input.text}
```

Return each concern as an item with:
  title    — the concern in under 10 words
  body     — why it matters here and what to do instead
  location — line number or function name
  severity — info | minor | major | critical
  tags     — short topic labels, e.g. ["correctness", "performance"]

Apply every lesson above. If a lesson says a class of comment was rejected,
do not produce that comment.
{self._closing()}"""

    def _closing(self) -> str:
        if self.thorough:
            return (
                "Aim for 3-6 comments. An empty list is only correct if the code is "
                "genuinely without flaw, which is rare."
            )
        return "Return an empty item list if you find nothing worth an engineer's time."

    # Phrases that make a rule read as suppression rather than as advice.
    _SUPPRESSES = (
        "do not flag", "don't flag", "avoid flagging", "never flag", "stop flagging",
        "do not raise", "don't raise", "do not comment on", "don't comment on",
        "do not report", "refrain from flagging", "no longer flag",
    )

    def to_insight(self, output, feedback: Feedback, task_input: TaskInput, item: Item | None = None):
        insight = super().to_insight(output, feedback, task_input, item)
        if feedback.action != "reject" or not insight.rule:
            return insight

        if not any(m in insight.rule.lower() for m in self._SUPPRESSES):
            # The model restated the concern as best-practice advice instead of
            # encoding the rejection. Stored as-is, that rule is injected under an
            # "avoid" marker while its text argues FOR the comment the engineer
            # discarded -- actively worse than learning nothing. Observed on the
            # first real review, so this correction is deterministic, not a prompt
            # hope: a 7B model will not reliably obey an instruction it has just
            # disobeyed.
            concern = (item.title if item else "this class of comment").rstrip(".")
            # Deliberately do NOT keep the drifted text, not even as a note:
            # render_lessons injects the rationale too, so an audit trail here
            # would smuggle the advice straight back into the next prompt. A test
            # caught exactly that.
            insight.rationale = "engineer rejected it; reflection drifted into advice and was rewritten"
            insight.rule = f'Do not flag "{concern}" in {task_input.scope.project} code.'

        if not feedback.note:
            # A rejection without a reason says the comment was unwanted but not
            # why, so the rule is a guess at scope. Start it low: if it is right it
            # gets reinforced, and if it is over-broad it decays out on its own.
            insight.confidence = min(insight.confidence, 0.35)
        return insight

    def reflect_prompt(self, task_input: TaskInput, item: Item, feedback: Feedback) -> str:
        header = {
            "accept": "The engineer ACCEPTED this comment and added a note.",
            "reject": "The engineer REJECTED this comment.",
            "edit": "The engineer KEPT the concern but REWROTE the comment.",
        }[feedback.action]
        guidance = {
            "accept": "Write a rule that reinforces finding this class of issue, and captures what made this comment useful.",
            "reject": (
                "Write a SUPPRESSION rule. It must instruct a future reviewer NOT to "
                "raise this class of comment, and must begin with 'Do not flag'. "
                "Example: 'Do not flag hardcoded paths in entry-point scripts; this "
                "project sets them deliberately.' "
                "Do NOT restate the concern as coding advice — that would argue FOR "
                "the very comment the engineer just threw out. Be precise about *when* "
                "it should not be raised; an over-broad rule suppresses real findings later."
            ),
            "edit": "Preserve the concern that was correct and encode the correction. The rule should describe how to phrase or scope this kind of comment properly.",
        }[feedback.action]

        parts = [
            header,
            "",
            "ORIGINAL COMMENT",
            f"title: {item.title}",
            f"body: {item.body}",
            f"severity: {item.severity}  tags: {', '.join(item.tags)}",
        ]
        if feedback.note:
            parts += ["", f"ENGINEER'S NOTE: {feedback.note}"]
        if feedback.action == "edit" and feedback.edited_body:
            parts += ["", f"ENGINEER'S CORRECTED VERSION: {feedback.edited_body}"]
        parts += [
            "",
            f"CONTEXT: language={task_input.scope.language}, framework={task_input.scope.framework}, project={task_input.scope.project}",
            "",
            guidance,
            "",
            "Also set scope_hint: 'any' if this rule holds for any codebase, or "
            f"'{task_input.scope.project}' if it is specific to this project's conventions.",
        ]
        return "\n".join(parts)


def input_from_file(
    path: str | Path, project: str = "default", team: str = "any", max_chars: int | None = None
) -> TaskInput:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    lang = _EXT_LANG.get(p.suffix.lower(), "any")
    framework = _guess_framework(text)

    numbered = _with_line_numbers(text)
    truncated = 0
    if max_chars and len(numbered) > max_chars:
        # Reviewing the first N characters well beats sending a 2,000-line module
        # past the context window and waiting minutes for a worse answer. The
        # caller is told, so a partial review is never mistaken for a full one.
        truncated = len(numbered) - max_chars
        numbered = numbered[:max_chars].rsplit("\n", 1)[0]

    return TaskInput(
        text=numbered,
        summary=p.name,
        scope=Scope(project=project, language=lang, framework=framework, team=team),
        meta={"path": str(p), "truncated_chars": truncated, "total_lines": text.count("\n") + 1},
    )


def input_from_text(
    text: str,
    summary: str = "snippet",
    language: str = "python",
    project: str = "default",
    framework: str = "any",
    team: str = "any",
) -> TaskInput:
    return TaskInput(
        text=_with_line_numbers(text),
        summary=summary,
        scope=Scope(project=project, language=language, framework=framework, team=team),
    )


def _with_line_numbers(text: str) -> str:
    return "\n".join(f"{n:>4} | {line}" for n, line in enumerate(text.splitlines(), 1))


def _guess_framework(text: str) -> str:
    low = text.lower()
    for needle, name in (
        ("pyspark", "pyspark"),
        ("from pyspark", "pyspark"),
        ("import pandas", "pandas"),
        ("fastapi", "fastapi"),
        ("django", "django"),
        ("flask", "flask"),
        ("react", "react"),
        ("airflow", "airflow"),
        ("dbt", "dbt"),
    ):
        if needle in low:
            return name
    return "any"
