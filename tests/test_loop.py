"""The tests that matter are the ones proving the agent *evolves*:

  * a rejection in review 1 changes the output of review 2 (test_lesson_changes_next_review)
  * the same lesson learned twice reinforces instead of duplicating (test_merge)
  * a contradicted rule decays and retires itself (test_decay)
  * a rule scoped to one project doesn't leak into another (test_scope_isolation)
  * the graph really pauses and resumes across process-level state (test_interrupt)
"""

from __future__ import annotations

import pytest

from selfevolve import (
    Config,
    ExperienceStore,
    Feedback,
    Insight,
    Item,
    RuleOut,
    Scope,
    SelfEvolvingAgent,
)
from selfevolve.llm import FakeProvider
from selfevolve.tasks.code_review import CodeReviewTask, input_from_text

CODE = """def ratio(a, b):
    return a / b
"""


@pytest.fixture
def cfg(tmp_path):
    c = Config(data_dir=tmp_path / "store")
    c.llm_backend = "fake"
    c.embed_backend = "hash"
    return c


@pytest.fixture
def agent(cfg):
    a = SelfEvolvingAgent(CodeReviewTask(), cfg=cfg, llm=FakeProvider(cfg))
    yield a
    a.close()


def test_interrupt_pauses_before_feedback(agent):
    ti = input_from_text(CODE, summary="ratio.py", project="p1")
    pending = agent.start(ti, thread_id="t1")
    assert pending["awaiting_feedback"] is True
    assert len(pending["items"]) == 2
    # the episode is durably paused, not lost
    assert agent.pending("t1") is not None


def test_feedback_produces_scoped_insight(agent):
    ti = input_from_text(CODE, summary="ratio.py", project="p1")
    pending = agent.start(ti, thread_id="t2")
    doc_item = next(i for i in pending["items"] if "docstring" in i.title.lower())
    result = agent.submit_feedback(
        [Feedback(item_id=doc_item.id, action="reject", note="our linter handles docstrings")],
        thread_id="t2",
    )
    assert result["counts"]["reject"] == 1
    assert result["learned"]
    assert result["trajectory_id"]
    assert agent.pending("t2") is None  # resumed to completion


def test_lesson_changes_next_review(agent, cfg):
    """The whole point: review 2 behaves differently because of review 1."""
    ti = input_from_text(CODE, summary="ratio.py", project="p1")
    pending = agent.start(ti, thread_id="a1")
    titles_before = {i.title for i in pending["items"]}
    assert any("docstring" in t.lower() for t in titles_before)

    doc_item = next(i for i in pending["items"] if "docstring" in i.title.lower())
    # Hand-write the rule the reflection step would produce, so the test asserts
    # retrieval + injection rather than the fake provider's prose.
    agent.llm.scripted = [
        {
            "rule": "Do not flag missing docstring comments; the team's linter enforces them.",
            "rationale": "Rejected as noise.",
            "scope_hint": "any",
        }
    ]
    agent.submit_feedback(
        [Feedback(item_id=doc_item.id, action="reject", note="linter covers it")], thread_id="a1"
    )

    ti2 = input_from_text("def scale(x, y):\n    return x / y\n", summary="scale.py", project="p1")
    pending2 = agent.start(ti2, thread_id="a2")
    used = [i.rule for i in pending2["retrieved_insights"]]
    assert any("docstring" in r.lower() for r in used), "lesson was not retrieved"
    titles_after = {i.title.lower() for i in pending2["items"]}
    assert not any("docstring" in t for t in titles_after), "lesson was retrieved but ignored"
    assert any("division" in t for t in titles_after), "the real bug got suppressed too"


def test_merge_reinforces_instead_of_duplicating(cfg):
    store = ExperienceStore(cfg)
    scope = Scope(project="p1", language="python")
    a = store.add_insight(Insight(rule="Do not flag missing docstrings", scope=scope, origin_action="reject"))
    b = store.add_insight(Insight(rule="do not flag   Missing Docstrings", scope=scope, origin_action="reject"))
    assert a.id == b.id
    assert b.support == 2
    assert b.confidence > a.confidence
    assert len(store.all_insights()) == 1


def test_decay_and_retirement(cfg):
    store = ExperienceStore(cfg)
    scope = Scope(project="p1", language="python")
    store.add_insight(Insight(rule="Always flag broad excepts", scope=scope, origin_action="reject", confidence=0.6))
    for _ in range(4):  # later feedback keeps contradicting it
        store.add_insight(Insight(rule="Always flag broad excepts", scope=scope, origin_action="edit"))
    live = store.all_insights()
    assert live == [] or live[0].retired
    assert store.retrieve_insights("broad except handling", scope=scope) == []


def test_scope_isolation(cfg):
    store = ExperienceStore(cfg)
    store.add_insight(
        Insight(rule="Prefer broadcast joins under 10MB", scope=Scope(project="etl", language="python", framework="pyspark"))
    )
    assert store.retrieve_insights("join two dataframes", scope=Scope(project="etl", language="python", framework="pyspark"))
    assert not store.retrieve_insights("join two dataframes", scope=Scope(project="webapp", language="typescript"))


def test_unexplained_accept_mints_no_rule(agent):
    ti = input_from_text(CODE, summary="ratio.py", project="p1")
    pending = agent.start(ti, thread_id="acc")
    result = agent.submit_feedback(
        [Feedback(item_id=i.id, action="accept") for i in pending["items"]], thread_id="acc"
    )
    assert result["learned"] == []
    assert result["counts"]["accept"] == 2


def test_edit_preserves_concern_and_encodes_correction(agent, cfg):
    ti = input_from_text(CODE, summary="ratio.py", project="p1")
    pending = agent.start(ti, thread_id="ed")
    div = next(i for i in pending["items"] if "division" in i.title.lower())
    result = agent.submit_feedback(
        [
            Feedback(
                item_id=div.id,
                action="edit",
                edited_body="Guard the denominator or let it raise deliberately — say which.",
                note="concern is right, the fix wasn't specific enough",
            )
        ],
        thread_id="ed",
    )
    assert result["counts"]["edit"] == 1
    learned = result["learned"]
    assert learned and learned[0].origin_action == "edit"
    # an edit must not suppress the concern the way a rejection does
    assert "do not flag" not in learned[0].rule.lower()


def test_feedback_on_unknown_item_is_dropped(agent):
    ti = input_from_text(CODE, summary="ratio.py", project="p1")
    agent.start(ti, thread_id="bad")
    result = agent.submit_feedback(
        [Feedback(item_id="item_does_not_exist", action="reject", note="x")], thread_id="bad"
    )
    assert result["learned"] == []
    assert sum(result["counts"].values()) == 0


def test_metrics_track_false_positive_rate(agent, cfg):
    ti = input_from_text(CODE, summary="ratio.py", project="p1")
    pending = agent.start(ti, thread_id="m1")
    agent.submit_feedback(
        [
            Feedback(item_id=pending["items"][0].id, action="accept"),
            Feedback(item_id=pending["items"][1].id, action="reject", note="noise"),
        ],
        thread_id="m1",
    )
    m = ExperienceStore(cfg).metrics()
    assert m.reviews == 1 and m.items == 2
    assert m.acceptance_rate == 0.5 and m.false_positive_rate == 0.5


def test_thorough_mode_changes_the_ask(cfg):
    """Two modes with genuinely different instructions, not a cosmetic flag.

    Default is tuned against false positives; thorough is tuned to produce
    something to rule on. A store with no rules cannot teach anything, so the
    bootstrapping case needs its own prompt.
    """
    ti = input_from_text(CODE, project="p1")

    default = CodeReviewTask()
    thorough = CodeReviewTask(thorough=True)

    assert "do not invent problems" in default.system_prompt()
    assert "Prefer 3-6 comments over" in thorough.system_prompt()

    assert "empty item list" in default.act_prompt(ti, [], [])
    assert "Aim for 3-6 comments" in thorough.act_prompt(ti, [], [])

    # both still defer to learned rules -- thoroughness must not override a
    # lesson that says a class of comment is noise
    for task in (default, thorough):
        assert "override your defaults" in task.system_prompt()
        assert "do not produce that comment" in task.act_prompt(ti, [], [])


def test_rejection_never_becomes_advice_arguing_for_the_comment(cfg):
    """The single worst failure mode this loop has, caught on a real review.

    A comment titled "Hardcoded path" was rejected with no note. Reflection wrote
    'Avoid hardcoding file paths... use configuration files' -- best-practice
    advice. Stored under an "avoid" marker, that rule reads as ARGUING FOR the
    very comment the engineer discarded, so the next review flags it harder.
    Learning nothing would have been better.
    """
    task = CodeReviewTask()
    ti = input_from_text(CODE, project="control")
    item = Item(title="Hardcoded path", body="The path is hardcoded.", tags=["flexibility"])
    drifted = RuleOut(
        rule=(
            "Avoid hardcoding file paths or resource locations in the code. Use "
            "configuration files or environment variables to specify paths."
        ),
        rationale="",
        scope_hint="control",
    )

    ins = task.to_insight(drifted, Feedback(item_id=item.id, action="reject"), ti, item)

    assert ins.rule.lower().startswith("do not flag"), ins.rule
    assert "Hardcoded path" in ins.rule
    assert "drifted into advice" in ins.rationale, "should say why it was rewritten"
    # a rejection with no stated reason is a guess at scope -- start it low
    assert ins.confidence <= 0.35

    rendered = task.render_lessons([ins], [])
    assert "[avoid" in rendered
    assert "Use configuration files" not in rendered, "advice must not reach the prompt"


def test_a_properly_phrased_suppression_is_left_alone(cfg):
    """The guard must not mangle a rule the model got right."""
    task = CodeReviewTask()
    ti = input_from_text(CODE, project="control")
    item = Item(title="Hardcoded path", body="...")
    good = RuleOut(
        rule="Do not flag hardcoded paths in entry-point scripts; they are set deliberately.",
        rationale="team convention",
        scope_hint="control",
    )
    fb = Feedback(item_id=item.id, action="reject", note="entry points set them on purpose")
    ins = task.to_insight(good, fb, ti, item)

    assert ins.rule == good.rule, "a correct suppression rule should survive untouched"
    assert ins.confidence > 0.35, "a rejection WITH a reason is stronger evidence"
