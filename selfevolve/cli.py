"""Terminal driver for the loop.

  selfevolve doctor                 prove the whole loop runs with the network cut
  selfevolve review FILE            review a file, then give a verdict per comment
  selfevolve insights               list learned rules
  selfevolve insights --retire ID   retire a rule that has gone stale
  selfevolve metrics                acceptance / rejection / edit rates
  selfevolve export                 dump everything learned to JSON
  selfevolve reset                  wipe the experience store
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from .config import Config
from .graph import SelfEvolvingAgent
from .llm import FakeProvider, OllamaProvider, get_provider
from .models import Feedback, Scope
from .offline import airgap, report
from .store import ExperienceStore
from .tasks.code_review import CodeReviewTask, input_from_file, input_from_text

BOLD, DIM, RED, GRN, YEL, CYN, RST = (
    "\033[1m",
    "\033[2m",
    "\033[31m",
    "\033[32m",
    "\033[33m",
    "\033[36m",
    "\033[0m",
)


def _agent(cfg: Config) -> SelfEvolvingAgent:
    llm = FakeProvider(cfg) if cfg.llm_backend == "fake" else get_provider(cfg)
    return SelfEvolvingAgent(CodeReviewTask(), cfg=cfg, llm=llm)


def cmd_review(args, cfg: Config) -> int:
    agent = _agent(cfg)
    ti = input_from_file(args.file, project=args.project, team=args.team)
    thread_id = args.thread or f"review-{uuid.uuid4().hex[:8]}"

    print(f"\n{BOLD}Reviewing{RST} {ti.summary}  {DIM}[{ti.scope.key()}]  thread={thread_id}{RST}")
    pending = agent.start(ti, thread_id)

    used = pending["retrieved_insights"]
    if used:
        print(f"\n{CYN}Retrieved {len(used)} lesson(s) from past reviews:{RST}")
        for ins in used:
            print(f"  {DIM}·{RST} {ins.rule}  {DIM}(conf {ins.confidence:.2f}, {ins.support}x){RST}")
    else:
        print(f"\n{DIM}No prior experience for this scope — reviewing from first principles.{RST}")

    items = pending["items"]
    if not items:
        print(f"\n{GRN}No comments.{RST} Nothing to learn from this round.")
        return 0

    print(f"\n{BOLD}{len(items)} comment(s){RST}\n")
    feedback: list[Feedback] = []
    for n, item in enumerate(items, 1):
        colour = {"info": DIM, "minor": YEL, "major": RED, "critical": RED}[item.severity]
        print(f"{BOLD}[{n}/{len(items)}]{RST} {colour}{item.severity}{RST}  {BOLD}{item.title}{RST}")
        if item.location:
            print(f"     {DIM}{item.location}{RST}")
        print(f"     {item.body}")
        if item.tags:
            print(f"     {DIM}tags: {', '.join(item.tags)}{RST}")

        action = ""
        while action not in ("a", "r", "e", "s"):
            action = (
                input(f"     {BOLD}[a]{RST}ccept  {BOLD}[r]{RST}eject  {BOLD}[e]{RST}dit  {BOLD}[s]{RST}kip > ")
                .strip()
                .lower()
                or "s"
            )
        if action == "s":
            print()
            continue

        edited = ""
        if action == "e":
            edited = input("     corrected comment > ").strip()
        note = input(f"     why? {DIM}(optional but this is what makes the lesson reusable){RST} > ").strip()
        feedback.append(
            Feedback(
                item_id=item.id,
                action={"a": "accept", "r": "reject", "e": "edit"}[action],
                edited_body=edited,
                note=note,
            )
        )
        print()

    if not feedback:
        print(f"{DIM}No verdicts given — nothing learned. The episode stays paused; "
              f"resume it with --thread {thread_id}.{RST}")
        return 0

    print(f"{DIM}Reflecting…{RST}")
    result = agent.submit_feedback(feedback, thread_id)
    c = result["counts"]
    print(
        f"\n{GRN}Stored{RST} trajectory {result['trajectory_id']}  "
        f"({c.get('accept',0)} accepted, {c.get('reject',0)} rejected, {c.get('edit',0)} edited)"
    )
    if result["learned"]:
        print(f"\n{BOLD}Learned:{RST}")
        for ins in result["learned"]:
            print(f"  {GRN}+{RST} {ins.rule}")
            if ins.rationale:
                print(f"    {DIM}{ins.rationale}{RST}")
            print(f"    {DIM}scope: {ins.scope.key()}  confidence: {ins.confidence:.2f}{RST}")
    else:
        print(f"{DIM}No new rules (plain accepts confirm existing behaviour).{RST}")
    agent.close()
    return 0


def cmd_insights(args, cfg: Config) -> int:
    store = ExperienceStore(cfg)
    if args.retire:
        store.retire_insight(args.retire)
        print(f"retired {args.retire}")
        return 0
    if args.delete:
        store.delete_insight(args.delete)
        print(f"deleted {args.delete}")
        return 0
    rows = store.all_insights(include_retired=args.all)
    if not rows:
        print("No insights yet. Run a review and give feedback.")
        return 0
    for ins in rows:
        flag = f"{DIM}(retired){RST} " if ins.retired else ""
        mark = {"accept": GRN + "+" + RST, "reject": RED + "-" + RST, "edit": YEL + "~" + RST}[
            ins.origin_action
        ]
        print(f"{mark} {flag}{BOLD}{ins.rule}{RST}")
        print(
            f"  {DIM}{ins.id}  scope={ins.scope.key()}  conf={ins.confidence:.2f}  "
            f"support={ins.support}  contradictions={ins.contradictions}{RST}"
        )
    return 0


def cmd_metrics(args, cfg: Config) -> int:
    store = ExperienceStore(cfg)
    scope = Scope(project=args.project) if args.project != "any" else None
    m = store.metrics(scope)
    print(f"\n{BOLD}Experience{RST}  {DIM}scope={args.project}{RST}")
    print(f"  reviews         {m.reviews}")
    print(f"  comments        {m.items}")
    print(f"  accepted        {m.accepted}")
    print(f"  rejected        {m.rejected}")
    print(f"  edited          {m.edited}")
    print(f"  insights        {m.insights}")
    print(f"\n  acceptance rate      {m.acceptance_rate:.0%}")
    print(f"  false-positive rate  {m.false_positive_rate:.0%}")
    print(
        f"\n{DIM}A rising insight count only proves writes are landing. Falling "
        f"false-positive rate across reviews is the thing to watch.{RST}"
    )
    return 0


def cmd_doctor(args, cfg: Config) -> int:
    """Verify offline operation by doing it, not by claiming it.

    Runs the entire loop — retrieve, act, human verdict, reflect, persist — with
    a socket guard that raises on any connection leaving this machine. If a
    dependency tries to phone home, this exits non-zero.
    """
    info = report()
    print(f"\n{BOLD}selfevolve doctor{RST}\n")
    print(f"  data dir       {info['data_dir']}")
    print(f"  database       {info['db']}")
    print(f"  llm            {info['llm_backend']} · {info['llm_model']}")
    print(f"  embeddings     {info['embed_backend']} · {info['embed_model']}")

    host_ok = info["ollama_is_local"]
    mark = f"{GRN}local{RST}" if host_ok else f"{RED}NOT LOCAL{RST}"
    print(f"  ollama host    {info['ollama_host']}  [{mark}]")

    if info["leaked_keys"]:
        print(f"\n  {RED}tracing keys still in env:{RST} {', '.join(info['leaked_keys'])}")

    probe = OllamaProvider(cfg)
    if probe.available():
        models = probe.models()
        print(f"\n  {GRN}Ollama reachable{RST} — {len(models)} model(s) installed")
        for m in models[:8]:
            print(f"    {DIM}·{RST} {m}")
        need = [cfg.llm_model, cfg.embed_model]
        missing = [n for n in need if not any(m.split(":")[0] == n.split(":")[0] for m in models)]
        if missing:
            print(f"\n  {YEL}not installed:{RST} {', '.join(missing)}")
            print(f"    {DIM}ollama pull {' && ollama pull '.join(missing)}{RST}")
    else:
        print(f"\n  {YEL}Ollama not reachable{RST} at {cfg.ollama_host}")
        print(f"    {DIM}the loop still runs — hashed embeddings and --backend fake{RST}")

    # --- the actual proof ---
    print(f"\n{BOLD}Airgap test{RST}  {DIM}(all non-loopback sockets blocked){RST}")
    probe_cfg = Config(data_dir=cfg.data_dir)
    probe_cfg.llm_backend = "fake"
    probe_cfg.embed_backend = "hash" if not host_ok or not probe.available() else cfg.embed_backend

    try:
        with airgap(allow_loopback=True):
            agent = SelfEvolvingAgent(
                CodeReviewTask(), cfg=probe_cfg, llm=FakeProvider(probe_cfg)
            )
            ti = input_from_text("def f(a, b):\n    return a / b\n", summary="doctor-probe")
            thread = f"doctor-{uuid.uuid4().hex[:8]}"
            pending = agent.start(ti, thread)
            print(f"  {GRN}✓{RST} retrieve + act        {len(pending['items'])} item(s)")
            result = agent.submit_feedback(
                [
                    Feedback(
                        item_id=pending["items"][0].id,
                        action="reject",
                        note="doctor probe — safe to delete",
                    )
                ],
                thread,
            )
            print(f"  {GRN}✓{RST} interrupt + resume    trajectory {result['trajectory_id']}")
            print(f"  {GRN}✓{RST} reflect + persist     {len(result['learned'])} rule(s)")

            store = ExperienceStore(probe_cfg)
            hits = store.retrieve_insights("division by zero", scope=ti.scope)
            print(f"  {GRN}✓{RST} retrieval             {len(hits)} rule(s) matched")

            # clean up after ourselves — a doctor run should leave no rules behind
            for ins in result["learned"]:
                store.delete_insight(ins.id)
            store.conn.execute("DELETE FROM trajectories WHERE id=?", (result["trajectory_id"],))
            store.conn.commit()
            store.close()
            agent.close()
    except Exception as exc:  # noqa: BLE001
        print(f"\n  {RED}✗ FAILED{RST}  {type(exc).__name__}: {exc}")
        print(f"\n{RED}Something tried to leave this machine, or the loop broke.{RST}")
        return 1

    print(f"\n{GRN}Full loop completed with the network cut.{RST}")
    print(f"{DIM}Nothing in this system requires an internet connection.{RST}\n")
    return 0


def cmd_export(args, cfg: Config) -> int:
    """Your learned rules are yours. One JSON file, no lock-in."""
    store = ExperienceStore(cfg)
    payload = {
        "version": 1,
        "insights": [i.model_dump() for i in store.all_insights(include_retired=True)],
        "trajectories": [t.model_dump() for t in store.all_trajectories()],
        "metrics": store.metrics().model_dump(),
    }
    text = json.dumps(payload, indent=2, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out} ({len(payload['insights'])} rules, "
              f"{len(payload['trajectories'])} trajectories)")
    else:
        print(text)
    return 0


def cmd_reset(args, cfg: Config) -> int:
    if input("Wipe all learned experience? [y/N] ").strip().lower() != "y":
        return 1
    ExperienceStore(cfg).reset()
    print("experience store cleared")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="selfevolve", description=__doc__)
    p.add_argument("--backend", choices=["ollama", "fake"], default=None)
    p.add_argument("--model", default=None, help="Ollama model tag, e.g. qwen3:8b")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("review")
    r.add_argument("file")
    r.add_argument("--project", default="default")
    r.add_argument("--team", default="any")
    r.add_argument("--thread", default=None)
    r.set_defaults(fn=cmd_review)

    i = sub.add_parser("insights")
    i.add_argument("--all", action="store_true", help="include retired rules")
    i.add_argument("--retire", default=None, metavar="ID")
    i.add_argument("--delete", default=None, metavar="ID")
    i.set_defaults(fn=cmd_insights)

    m = sub.add_parser("metrics")
    m.add_argument("--project", default="any")
    m.set_defaults(fn=cmd_metrics)

    d = sub.add_parser("doctor", help="prove the loop runs with the network cut")
    d.set_defaults(fn=cmd_doctor)

    e = sub.add_parser("export", help="dump everything learned to JSON")
    e.add_argument("--out", default=None, metavar="FILE")
    e.set_defaults(fn=cmd_export)

    s = sub.add_parser("reset")
    s.set_defaults(fn=cmd_reset)

    args = p.parse_args(argv)
    cfg = Config()
    if args.backend:
        cfg.llm_backend = args.backend
    if args.model:
        cfg.llm_model = args.model
    try:
        return args.fn(args, cfg)
    except KeyboardInterrupt:
        print("\naborted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
