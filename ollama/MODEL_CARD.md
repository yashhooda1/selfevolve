# selfevolve-reviewer

A `qwen2.5:7b-instruct` variant preloaded with the system prompt and decoding
settings from [selfevolve](https://github.com/yashhooda1/selfevolve) — a
code-review agent that learns from an engineer's corrections and stores the
lessons locally.

```
ollama run hoodarunner/selfevolve-reviewer
```

Paste in a function and it reviews it: a small number of high-signal comments,
each with a concrete concern and a specific recommendation. It is told not to
restate what the code does, not to flag formatting a linter already handles, and
to return nothing when the code is fine.

## What this model is not

**It does not learn.** The learning lives in the application, not in these
weights. `selfevolve` keeps accepted, rejected and edited review comments in a
local SQLite file, distils them into scoped rules, and injects the relevant ones
into the next review. A model on a public registry cannot contain anyone's
accumulated judgement — and shouldn't, since those rules are specific to one
person's codebases.

**Pointing `selfevolve` at this model changes nothing.** The agent sends its own
system message on every request, which takes precedence over the `SYSTEM` baked
into a Modelfile. This exists so the review prompt can be tried in one command,
with no Python and no clone.

## Why instruct rather than reasoning

Under a JSON-schema constraint a reasoning model's thinking phase has nowhere to
go. On a real review, `qwen3:8b` timed out at 180 seconds where
`qwen2.5:7b-instruct` finished the same file in 22. Reasoning quality is not the
bottleneck for this task: the model supplies general code knowledge, and the
learned rules supply the judgement.

## The full loop

```
retrieve lessons ─► review ─► human accepts/rejects/edits ─► reflect ─► persist
       ▲                                                                  │
       └────────────────── one SQLite file ◄──────────────────────────────┘
```

Runs entirely offline — one runtime dependency, local Ollama for generation and
embeddings, and CI that proves the whole loop inside a network namespace with no
interfaces.

[Source, tests and setup →](https://github.com/yashhooda1/selfevolve)

MIT.
