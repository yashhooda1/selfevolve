"""Streamlit feedback interface.

Three tabs:
  Review    paste or upload code, see the comments, rule on each one
  Memory    browse, edit, retire and delete learned rules (rule maintenance is
            not optional at scale — teams change frameworks and standards)
  Metrics   acceptance / rejection / edit rates over time

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Harden the environment BEFORE streamlit is imported — it reads its telemetry
# settings at import time, so the order here is load-bearing, not cosmetic.
from selfevolve.offline import harden_env  # noqa: E402

harden_env()

import streamlit as st  # noqa: E402

from selfevolve import Config, ExperienceStore, Feedback, Scope, SelfEvolvingAgent  # noqa: E402
from selfevolve.llm import FakeProvider, OllamaProvider  # noqa: E402
from selfevolve.tasks.code_review import CodeReviewTask, input_from_text  # noqa: E402

st.set_page_config(page_title="Self-Evolving Review Agent", page_icon="🔁", layout="wide")

SAMPLE = '''def compute_ratio(numerator, denominator):
    return numerator / denominator


def load_users(conn, ids):
    out = []
    for i in ids:
        out.append(conn.execute("SELECT * FROM users WHERE id = " + str(i)).fetchone())
    return out
'''


@st.cache_resource(show_spinner=False)
def get_agent(backend: str, model: str) -> SelfEvolvingAgent:
    cfg = Config()
    cfg.llm_backend = backend
    cfg.llm_model = model
    llm = FakeProvider(cfg) if backend == "fake" else OllamaProvider(cfg)
    return SelfEvolvingAgent(CodeReviewTask(), cfg=cfg, llm=llm)


@st.cache_resource(show_spinner=False)
def get_store() -> ExperienceStore:
    return ExperienceStore(Config())


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.title("🔁 selfevolve")
    st.caption("Retrieve → Act → Human verdict → Reflect → Persist")
    backend = st.selectbox("LLM backend", ["ollama", "fake"], index=0)
    model = st.text_input("Model", value=Config().llm_model, disabled=backend == "fake")
    st.divider()
    project = st.text_input("Project", value="default")
    language = st.selectbox("Language", ["python", "sql", "typescript", "scala", "go", "any"])
    framework = st.text_input("Framework", value="any")
    team = st.text_input("Team", value="any")
    scope = Scope(project=project, language=language, framework=framework, team=team)
    st.caption(f"Rules learned here are filed under `{scope.key()}`.")

agent = get_agent(backend, model)
store = get_store()

tab_review, tab_memory, tab_metrics = st.tabs(["Review", "Memory", "Metrics"])

# ------------------------------------------------------------------- review

with tab_review:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    if "pending" not in st.session_state:
        st.session_state.pending = None
    if "result" not in st.session_state:
        st.session_state.result = None

    left, right = st.columns([3, 2])

    with left:
        uploaded = st.file_uploader("Upload a file", type=None, label_visibility="collapsed")
        code = (
            uploaded.read().decode("utf-8", errors="replace")
            if uploaded
            else st.text_area("Code", value=SAMPLE, height=280, label_visibility="collapsed")
        )
        if st.button("Review", type="primary"):
            st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
            st.session_state.result = None
            ti = input_from_text(
                code,
                summary=uploaded.name if uploaded else "snippet",
                language=language,
                project=project,
                framework=framework,
                team=team,
            )
            with st.spinner("Retrieving experience and reviewing…"):
                st.session_state.pending = agent.start(ti, st.session_state.thread_id)

    with right:
        pending = st.session_state.pending
        if pending:
            used = pending["retrieved_insights"]
            st.subheader(f"Context injected ({len(used)} lesson{'' if len(used)==1 else 's'})")
            if used:
                for ins in used:
                    icon = {"accept": "✅", "reject": "🚫", "edit": "✏️"}[ins.origin_action]
                    st.markdown(
                        f"{icon} {ins.rule}  \n"
                        f"<span style='opacity:.6;font-size:.85em'>conf {ins.confidence:.2f} · "
                        f"seen {ins.support}× · {ins.scope.key()}</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Nothing learned for this scope yet. This review is from first principles.")

    st.divider()

    pending = st.session_state.pending
    if pending and st.session_state.result is None:
        items = pending["items"]
        if not items:
            st.success("No comments. Nothing to learn from this round.")
        else:
            st.subheader(f"{len(items)} comment{'' if len(items)==1 else 's'} awaiting your verdict")
            st.caption(
                "The graph is paused at an interrupt. Nothing is learned until you rule on these."
            )
            with st.form("feedback_form"):
                verdicts = {}
                for n, item in enumerate(items, 1):
                    sev = {"info": "🔵", "minor": "🟡", "major": "🟠", "critical": "🔴"}[item.severity]
                    with st.container(border=True):
                        st.markdown(f"**{sev} {item.title}**  ·  `{item.location or '—'}`")
                        st.write(item.body)
                        if item.tags:
                            st.caption(" · ".join(item.tags))
                        c1, c2 = st.columns([1, 2])
                        action = c1.radio(
                            "Verdict",
                            ["skip", "accept", "reject", "edit"],
                            key=f"a{n}",
                            horizontal=False,
                        )
                        note = c2.text_input(
                            "Why? (this is what makes the lesson reusable)", key=f"n{n}"
                        )
                        edited = c2.text_area(
                            "Corrected comment (for edit)", key=f"e{n}", height=68
                        )
                        verdicts[item.id] = (action, note, edited)
                submitted = st.form_submit_button("Submit feedback", type="primary")

            if submitted:
                fb = [
                    Feedback(item_id=iid, action=a, note=nt, edited_body=ed)
                    for iid, (a, nt, ed) in verdicts.items()
                    if a != "skip"
                ]
                if not fb:
                    st.warning("No verdicts — the episode stays paused.")
                else:
                    with st.spinner("Reflecting and persisting…"):
                        st.session_state.result = agent.submit_feedback(
                            fb, st.session_state.thread_id
                        )
                    get_store.clear()
                    st.rerun()

    if st.session_state.result:
        r = st.session_state.result
        c = r["counts"]
        st.success(
            f"Stored trajectory `{r['trajectory_id']}` — "
            f"{c.get('accept',0)} accepted · {c.get('reject',0)} rejected · {c.get('edit',0)} edited"
        )
        if r["learned"]:
            st.subheader("Learned")
            for ins in r["learned"]:
                with st.container(border=True):
                    st.markdown(f"**{ins.rule}**")
                    if ins.rationale:
                        st.caption(ins.rationale)
                    st.caption(f"scope `{ins.scope.key()}` · confidence {ins.confidence:.2f}")
        else:
            st.caption("No new rules — unexplained accepts only confirm existing behaviour.")
        if st.button("New review"):
            st.session_state.pending = None
            st.session_state.result = None
            st.rerun()

# ------------------------------------------------------------------- memory

with tab_memory:
    store = get_store()
    show_retired = st.checkbox("Show retired rules", value=False)
    rows = store.all_insights(include_retired=show_retired)
    st.caption(f"{len(rows)} rule(s) in `review_insights`")
    if not rows:
        st.info("No rules yet. Review something and give feedback.")
    for ins in rows:
        icon = {"accept": "✅", "reject": "🚫", "edit": "✏️"}[ins.origin_action]
        label = ("~~" + ins.rule + "~~") if ins.retired else ins.rule
        with st.expander(f"{icon} {label}"):
            new_rule = st.text_area("Rule", value=ins.rule, key=f"r-{ins.id}", height=68)
            conf = st.slider("Confidence", 0.0, 1.0, float(ins.confidence), key=f"c-{ins.id}")
            st.caption(
                f"`{ins.id}` · scope `{ins.scope.key()}` · support {ins.support} · "
                f"contradictions {ins.contradictions}"
            )
            b1, b2, b3 = st.columns(3)
            if b1.button("Save", key=f"s-{ins.id}"):
                ins.rule, ins.confidence = new_rule, conf
                store.update_insight(ins)
                st.rerun()
            if b2.button("Retire", key=f"x-{ins.id}"):
                store.retire_insight(ins.id)
                st.rerun()
            if b3.button("Delete", key=f"d-{ins.id}"):
                store.delete_insight(ins.id)
                st.rerun()

    st.divider()
    st.subheader("Trajectories")
    for traj in store.all_trajectories()[:20]:
        c = traj.counts()
        st.markdown(
            f"`{traj.id}` **{traj.input_summary}** · {traj.scope.key()} · "
            f"{c['accept']}✅ {c['reject']}🚫 {c['edit']}✏️"
        )

# ------------------------------------------------------------------ metrics

with tab_metrics:
    m = get_store().metrics()
    a, b, c, d = st.columns(4)
    a.metric("Reviews", m.reviews)
    b.metric("Comments", m.items)
    c.metric("Acceptance rate", f"{m.acceptance_rate:.0%}")
    d.metric("False-positive rate", f"{m.false_positive_rate:.0%}")
    st.bar_chart(
        {"accepted": [m.accepted], "rejected": [m.rejected], "edited": [m.edited]}
    )
    st.caption(
        "A growing rule count only proves writes are landing. The number that tells you the "
        "agent is actually getting better is the false-positive rate falling across reviews "
        "of comparable code."
    )
