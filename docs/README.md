# CogriaAgent documentation

Start here if you are new: [Getting started](./getting-started.md) walks from an
empty directory to a working agent in about ten minutes.

| Document | What it covers |
|---|---|
| [Getting started](./getting-started.md) | Prerequisites, scaffolding, running the stack, first conversation |
| [Architecture](./architecture.md) | The three tiers, the injection seams, a request end to end, design decisions |
| [Integration guide](./integration-guide.md) | Exactly what your side delivers vs. what the framework provides; embedded vs. adapter mode |
| [Backend contract](../packages/contract/CONTRACT.md) | The language-agnostic HTTP spec every business backend implements |
| [Configuration](./configuration.md) | Every environment variable and config field, with defaults |
| [Attachments](./attachments.md) | File uploads: extraction, prompt injection format, token budget, security |
| [Roadmap](./roadmap.md) | Shipped, in progress, and planned |

## Concepts in one minute

**Action** — one capability of your product, declared as a name, an English
description, a JSON Schema for its parameters, and a handler. The description is
what the model reads to decide whether the action fits the user's request.

**Catalog** — the list of actions your backend advertises. The kernel fetches it
and synthesises one LLM tool per entry, so adding a capability never means
touching the framework.

**Propose / confirm** — any action marked `requires_confirm` runs in two phases:
a dry run that validates and returns a human-readable summary, then a real
execution gated on a one-shot token. The user approves in between.

**Artifact** — a front-end tool the model calls to render something in the side
panel (a chart, an image, an embedded page). It never executes server-side; the
tool call *is* the render instruction.

**Seam** — a Python `Protocol` the kernel depends on instead of a concrete
implementation. Seams are what keep the kernel free of business knowledge; see
[Architecture](./architecture.md#the-injection-seams).
