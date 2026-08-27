# selfevolve

[![airgap](https://github.com/yashhooda1/selfevolve/actions/workflows/airgap.yml/badge.svg)](https://github.com/yashhooda1/selfevolve/actions/workflows/airgap.yml)
[![tests](https://github.com/yashhooda1/selfevolve/actions/workflows/ci.yml/badge.svg)](https://github.com/yashhooda1/selfevolve/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10--3.14-blue)](pyproject.toml)
[![dependencies](https://img.shields.io/badge/runtime%20deps-pydantic-brightgreen)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

A self-evolving agent that runs entirely on your machine. No API keys, no cloud, no
telemetry, no network — verified by a test that blocks every outbound socket and runs
the whole loop anyway.

> The **airgap** badge is the one that matters. It's green only if the entire test suite
> and `selfevolve doctor` pass inside a Linux network namespace with no interfaces —
> no route to anywhere — and a fresh install pulls no HTTP client, telemetry SDK, or
> weights downloader. If a dependency grows a phone-home in a patch release, that badge
> goes red on the 1st of the month without anyone touching this repo.

Most agents can finish a task but can't learn from what happens afterwards. You correct
them, they agree, you start a new session, and the same mistake comes back. The model
usually isn't the problem — the application has nowhere to put the lesson.

`selfevolve` adds that place. It implements the ExpeL loop
([Zhao et al., 2023](https://arxiv.org/abs/2308.10144)): distil natural-language lessons
from past experience, retrieve them during related tasks, and let the improved context —
not changed weights — do the work.

```
retrieve ──► act ──► human verdict ──► reflect ──► persist
    ▲                                                 │
    └────────── experience.db (one file) ◄────────────┘
```

The reference implementation is a **code-review agent**: it reviews code, waits for you
to accept / reject / edit each comment, turns those verdicts into scoped rules, and
injects the relevant ones into the next review.

---

## Offline is a property, not a claim

```bash
selfevolve doctor
```

```
  ✓ retrieve + act        2 item(s)
  ✓ interrupt + resume    trajectory traj_0c7317a8c580
  ✓ reflect + persist     1 rule(s)
  ✓ retrieval             1 rule(s) matched

Full loop completed with the network cut.
```

That runs the real loop with a socket guard installed that raises on any connection to a
non-loopback address. It exits non-zero if anything tries to leave the machine. The test
suite runs the same guard around every test, so a `requests.get` added six months from
now fails the build instead of silently working on a connected laptop.

Three things had to change for that to be true, and each is worth knowing:

**No `sentence-transformers`.** It's a better embedder, but it fetches weights from
huggingface.co on first use. A build that downloads 130MB the first time you run it isn't
offline — it's offline *after* you've been online. Embeddings come from your local Ollama
instead (`nomic-embed-text`), with a stdlib hashed fallback so the loop still runs if
Ollama is down.

**No ChromaDB.** It's a local database, but it ships PostHog telemetry and an ONNX
runtime. Memory is now a single SQLite file with a brute-force cosine scan. At the scale
this actually operates — hundreds to low thousands of rules — a full scan is faster than
an index lookup, because there's no index to traverse.

**No LangGraph.** This was written on LangGraph first. `langgraph` → `langchain-core` →
`langsmith` → `requests` + `httpx`: a vendor tracing client sitting in the import graph of
a system whose premise is that it runs on your machine. Disabling it with an environment
variable is a runtime flag; not shipping it is a guarantee. What LangGraph provided here
was `interrupt()` and a checkpointer — for a five-node linear graph, that's a table and
about eighty lines.

**The dependency list is now: `pydantic`.** Streamlit is optional and only for the UI.
That's the entire audit surface.

---

## Quick start

```bash
pip install -e ".[all]"
selfevolve doctor                    # works before you pull anything
# if your shell can't find `selfevolve`, its scripts dir isn't on PATH —
# `python -m selfevolve.cli doctor` is identical and always works

ollama pull qwen3:8b                 # generation
ollama pull nomic-embed-text         # embeddings — one download, then offline forever

selfevolve review examples/sample_code/pipeline.py --project etl
streamlit run app/streamlit_app.py
```

No model yet? `--backend fake` exercises the entire loop — retrieval, the pause,
reflection, persistence — deterministically and with no service at all.

Then watch the loop close: review `pipeline.py`, reject a comment you think is noise with
a one-line reason, and review `examples/sample_code/handlers.py` — different file,
different names, same classes of issue. The rejected comment doesn't come back.

```bash
selfevolve insights                  # what it has learned
selfevolve metrics                   # acceptance / rejection / edit rates
selfevolve export --out lessons.json # your rules, portable, no lock-in
selfevolve insights --retire ins_abc123
```

---

## Two ordering rules that make this work

Everything else is plumbing. These two are the design:

**`retrieve` runs before `act`, unconditionally.** It's an edge in the state machine, not
a tool the model may choose to call. Every episode checks for relevant experience, so
behaviour never depends on the model remembering to look things up.

**`reflect` runs after the human verdict, never before.** The agent can't certify its own
output as correct — you supply that judgement. Reflection reads the original comment and
the verdict together, which is why *"rejected — our linter already enforces this"* becomes
a usable rule and a bare rejection doesn't.

Between them the loop genuinely stops. `start()` writes the episode to SQLite and returns;
the process can exit. `submit_feedback()` with the same `thread_id` — minutes or days
later, from the CLI or Streamlit — resumes exactly where it left off. `open_episodes()`
lists reviews you started and never ruled on.

---

## Memory is two tables, not one

| table | holds | injected as |
|---|---|---|
| `insights` | short reusable rules distilled from feedback | direct guidance |
| `trajectories` | whole episodes: input, comments, verdicts | worked examples |

The split matters. Merge them and retrieval hands you a 2 KB blob where you wanted a rule,
or a bare rule where you wanted to show the model what a similar review looked like.

Both live in `.selfevolve/experience.db` — along with paused episodes and the embedding
cache. **That one file is the entire system state.** `cp experience.db backup.db` is the
whole backup story; `sqlite3 experience.db` is the whole debugging story.

> The database deliberately does *not* use WAL mode. WAL is faster under concurrency but
> leaves recent commits in a sidecar `-wal` file, which quietly breaks that promise — a
> test caught exactly that. `ExperienceStore.backup()` uses SQLite's online backup API for
> a consistent copy while Streamlit has the file open.

---

## More memory is not better output

A rising rule count only proves writes are landing. Three things stop the store from
degrading into noise, and they're implemented, not left as an exercise:

**Scope.** Every rule carries `project / language / framework / team`. A rule learned on
your Spark ETL doesn't surface while reviewing TypeScript. `any` is the wildcard, so a
genuinely general lesson still applies everywhere.

**Merging.** Rules are fingerprinted on `(normalized text, scope)`. Learning the same
lesson twice raises `support` and `confidence` instead of writing a second row that
competes for the same slot in the injection budget.

**Decay.** A rule contradicted by later feedback loses half its confidence each time and
retires itself below a floor. Teams change frameworks; a rule that was right in March
should be able to die in September without anyone doing archaeology. The Memory tab also
lets you edit, retire, and delete by hand.

One more guard: an **accept with no note mints no rule.** Otherwise you accumulate a
hundred restatements of "finding real bugs is good."

---

## Plugging in a different domain

The loop is domain-independent. A `Task` answers five questions:

```python
class MyTask(BaseTask):
    name = "my_task"

    def system_prompt(self) -> str: ...
    def retrieval_text(self, task_input) -> str: ...          # how to index it
    def act_prompt(self, task_input, insights, trajectories) -> str: ...
    def reflect_prompt(self, task_input, item, feedback) -> str: ...
    # output_schema / to_items / to_insight have working defaults
```

Anything decomposable into items a human can accept, reject, or edit fits: writing agents
(suggested edits), research agents (claims), data-quality agents (flagged records).
`BaseTask` supplies the defaults; `CodeReviewTask` overrides four methods.

Swapping the model runtime is one method — implement
`structured(system, prompt, schema) -> Model`. `OllamaProvider` uses Ollama's
JSON-schema-constrained decoding (so the model *can't* emit malformed output) and strips
`<think>` blocks from reasoning models. Pointing it at llama.cpp's server or LM Studio is
a twenty-line class.

---

## Using it programmatically

```python
from selfevolve import SelfEvolvingAgent, Feedback
from selfevolve.tasks.code_review import CodeReviewTask, input_from_file

agent = SelfEvolvingAgent(CodeReviewTask())
pending = agent.start(input_from_file("etl/jobs/daily.py", project="etl"), thread_id="pr-482")

for item in pending["items"]:
    print(item.render())

agent.submit_feedback(
    [Feedback(item_id=pending["items"][0].id, action="reject",
              note="we allow bare excepts in the retry wrapper")],
    thread_id="pr-482",
)
```

`start()` returns and the process can die. `submit_feedback()` days later resumes the same
checkpointed episode.

---

## Layout

```
selfevolve/
  offline.py      env lockdown + the socket guard that enforces all of this
  models.py       Item, Feedback, Insight, Trajectory, Scope, Metrics
  store.py        SQLite: insights, trajectories, episodes, embedding cache
  embeddings.py   Ollama embeddings with a stdlib fallback
  llm.py          Ollama structured output via urllib + a deterministic fake
  task.py         the Task protocol — the framework seam
  graph.py        the checkpointed state machine
  cli.py          review / insights / metrics / export / doctor / reset
  tasks/code_review.py
app/streamlit_app.py   review / memory / metrics
tests/                 25 tests: the loop, and the offline guarantees
```

## Tests

```bash
pytest
```

25 tests. `test_loop.py` asserts the properties that make it an *evolving* agent: a
rejection in review 1 changes the output of review 2; a repeated lesson reinforces instead
of duplicating; a contradicted rule retires itself; a project-scoped rule doesn't leak;
the loop really pauses and resumes.

`test_offline.py` asserts the properties that make it a *local* one: the guard blocks
external connections and permits loopback; the full loop runs with even loopback cut; no
banned network library is in the import graph; no executable line names a remote host;
`doctor` exits zero and leaves no rules behind; and the database survives being copied to
another path.

## Where the database lives

Default is `./.selfevolve/experience.db`, relative to wherever you run the command.
Set `SELFEVOLVE_DATA_DIR` to pin it somewhere stable:

```bash
setx SELFEVOLVE_DATA_DIR "%USERPROFILE%\.selfevolve"     # Windows
export SELFEVOLVE_DATA_DIR=~/.selfevolve                  # macOS / Linux
```

**Do not put it inside OneDrive, Dropbox, iCloud, or Google Drive.** Those services copy
a file whenever it changes, with no idea that a database has a journal that has to stay
consistent with it — and OneDrive on Windows can hold a lock mid-upload, which surfaces as
intermittent `database is locked` errors that look like a bug in this program. `doctor`
warns you if it detects one of those folders.

---

## Configuration

Every setting is an environment variable — `SELFEVOLVE_LLM_BACKEND`,
`SELFEVOLVE_LLM_MODEL`, `SELFEVOLVE_EMBED_BACKEND`, `SELFEVOLVE_EMBED_MODEL`,
`SELFEVOLVE_DATA_DIR`, `SELFEVOLVE_TOP_K_INSIGHTS`, `SELFEVOLVE_MIN_CONFIDENCE`,
`SELFEVOLVE_RETIRE_BELOW`, `OLLAMA_HOST`. See `config.py`. Every default points at
something on this machine, and `doctor` tells you if you've overridden one to something
that isn't loopback.

## Publishing

Three different registries, three different artifacts. They are not interchangeable.

**GitHub — the code.** This is where the project lives.

```bash
gh repo create hoodarunner/selfevolve --public --source=. --remote=origin --push
```

**PyPI — the installable package.** So people can `pip install selfevolve` without cloning.

```bash
pip install build twine
python -m build
twine upload dist/*
```

**Ollama — a reviewer model, not the app.** ollama.com hosts model weights, so the
application cannot go there. What can is a `qwen3:8b` variant preloaded with the
reviewer's system prompt, for anyone who wants to try the review step with no install:

```bash
ollama create hoodarunner/selfevolve-reviewer -f ollama/Modelfile
ollama push hoodarunner/selfevolve-reviewer
```

See [`ollama/README.md`](ollama/README.md) for key setup and the important caveat: that
model does **not** carry your learned rules, and pointing `selfevolve` at it does not
change the agent's behaviour. Your rules live in `experience.db` on your machine and are
injected per request. A public registry can't hold your accumulated judgement — and
shouldn't.

That's the premise restated: the learning is in the application, not the weights.

---

## Honest limitations

**Hashed-fallback retrieval is weak.** Without Ollama, embeddings are hashed n-grams — a
lesson phrased very differently from the current code may not be retrieved. It exists so
the loop degrades instead of crashing, not as a real embedder. `ollama pull
nomic-embed-text` once and this goes away.

**Brute-force search has a ceiling.** Fine to a few thousand rules. Past that, swap the
ranking loop in `retrieve_insights` for sqlite-vec — it's the only place that scans.

**Rules compound in ways you won't predict.** Five rules about phrasing can collectively
push reviews into a voice you didn't intend, with no single rule at fault. Read
`selfevolve insights` occasionally as a whole, not just as a diff.

MIT.
