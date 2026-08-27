# Publishing the reviewer to Ollama

> **Namespaces differ.** The code lives at `github.com/yashhooda1/selfevolve`; the model
> lives at `ollama.com/hoodarunner/selfevolve-reviewer`. `hoodarunner` is the Ollama and
> HuggingFace handle, `yashhooda1` is the GitHub one. The `ollama create` name must match
> your **Ollama** username or the push is rejected.

Ollama's registry hosts **models**, not applications. `selfevolve` itself is a Python
package — it belongs on GitHub and PyPI. What can go to ollama.com is this: a `qwen2.5:7b-instruct`
variant preloaded with the reviewer's system prompt and decoding settings, so anyone can
run the review step directly:

```bash
ollama run hoodarunner/selfevolve-reviewer
```

## Publish

```bash
# 1. one-time: register your public key at ollama.com/settings/keys
#    Windows: C:\Users\<you>\.ollama\id_ed25519.pub
#    macOS/Linux: ~/.ollama/id_ed25519.pub
#    (generate with `ssh-keygen -t ed25519 -f ~/.ollama/id_ed25519` if absent)

# 2. build it locally — the model name MUST start with your ollama.com username
ollama create hoodarunner/selfevolve-reviewer -f ollama/Modelfile

# 3. try it before publishing
ollama run hoodarunner/selfevolve-reviewer "review this: def f(a,b): return a/b"

# 4. push
ollama push hoodarunner/selfevolve-reviewer
```

It lands at `https://ollama.com/hoodarunner/selfevolve-reviewer`. Re-running
`ollama push` after a change republishes it.

## The model card

The Modelfile format has no description or README field — `FROM`, `PARAMETER`,
`TEMPLATE`, `SYSTEM`, `ADAPTER`, `LICENSE`, `MESSAGE` and nothing else. The text on
the model's ollama.com page is edited on the website after the first push.

`MODEL_CARD.md` in this directory is that text, kept in version control so it can't
drift from the repo the way an untracked web form does. Paste it into the
description field on the model page after pushing.

## Point selfevolve at it

```bash
setx SELFEVOLVE_LLM_MODEL "hoodarunner/selfevolve-reviewer"   # Windows
export SELFEVOLVE_LLM_MODEL=hoodarunner/selfevolve-reviewer   # macOS/Linux
```

## What this does and doesn't do

**Does:** give people a one-command way to try the reviewer with no Python, no install,
no repo clone. Good for a README demo and for anyone who just wants to see the prompt's
behaviour.

**Doesn't:** change how `selfevolve` behaves. The agent sends its own system message on
every request, and that takes precedence over the `SYSTEM` baked into a Modelfile. It also
carries none of the learned rules — those live in `experience.db` on your machine and are
injected per request. A model on a public registry cannot contain your accumulated
lessons, and shouldn't: they're specific to your codebases and your judgement.

The learning is in the application, not the weights. That's the whole premise — swapping
the base model out changes the review quality, not the fact that it improves.

## Don't hand-edit `Modelfile`

It's generated from `CodeReviewTask.system_prompt()` by `scripts/build_modelfile.py`, and
a test fails if the two drift. Change the prompt in the task, then regenerate:

```bash
python scripts/build_modelfile.py
```
