# Roadmap

Honest status. Anything marked shipped has tests; anything marked planned is
design only, with no code behind it yet.

## Shipped

**Orchestration kernel.** LangGraph tool loop with a turn cap, FastAPI service,
custom SSE protocol, dynamic tool synthesis from a catalog, and the injection
seams that keep it domain-agnostic. Verified end to end against live models.

**HTTP contract and Python SDK.** A language-agnostic contract (catalog, action
envelope, auth exchange, propose/confirm) with pydantic types, JSON Schema, and a
conformance suite any backend can run. `backend-py` implements it: action base
class, registry discovery, one-shot params-bound proposal tokens, automatic
audit, envelope translation.

**Front-end.** Next.js 16 chat UI on assistant-ui with streaming, tool-call
visualisation, an inline confirm card, an artifact panel with three builtin
renderers (chart/table, image, image comparison), i18n, and the BFF that trades
a session cookie for a cached JWT.

**Scaffolding.** A zero-dependency Node CLI that generates a working agent —
backend actions, kernel wiring, Docker Compose, front-end configuration — from
one command.

**Persistence and attachments.** SQLite/Postgres conversation storage, plus
uploads for images, PDF, Word, Excel and PowerPoint with server-side extraction,
a token budget, and the security controls listed in
[Attachments](./attachments.md).

**Context management.** Three layers, cheapest first, all configurable in
[Configuration](./configuration.md): oversized tool results are trimmed as they
enter history (no model call); older messages are folded into an incremental
checkpoint once the *next* request would fill too much of the context window;
and a request the provider rejects as too long triggers a forced compaction and
one retry rather than surfacing a raw provider error. Compaction never deletes —
an optional `history_search` tool reads exact values back out of folded
messages.

**Resumable streams.** Navigating away mid-reply no longer kills generation or
stores a truncated message: the model run is decoupled from the SSE response and
the turn is persisted before `done` is emitted. Coming back re-attaches — the
kernel keeps a per-conversation registry of runs in flight with an event
backlog, so a returning client replays what it missed and then follows the rest
live. Conversations are owned by the JWT subject and every read, continuation
and replay is gated on it.

## Where this is going

Ordered by what unblocks what, not by dates — this is a small project and a date
in a public roadmap is a promise it cannot keep. Nothing below has code yet.

Each horizon states why it comes before the next one. If you disagree with the
ordering, that is a useful issue to open.

### 1 · Be installable

The one thing holding everything else back: today the only way to use CogriaAgent
is to clone it.

- Publish `cogria-agentserv`, `cogria-backend` and `cogria-contract` to PyPI, and
  `@cogria/agent-ui` to npm
- Semantic versioning, a changelog, and release automation in CI
- Version the HTTP contract explicitly, so a backend can declare which revision
  it implements and the conformance suite can check against it

*Why first:* a framework you cannot install is a framework nobody adopts. It also
forces anyone who wants it to copy the source, and a copy diverges from the day
it is made — the fix for that is a versioned dependency, which needs a release.

### 2 · Earn production trust

The questions an operations team asks before anything reaches real users.

- **Cost governance.** Per-turn token accounting written back to storage, budgets
  enforced before a request runs, a quota endpoint contract, a kill switch, and a
  usage banner in the UI. `GET /conversations/{id}` already reports
  `estimated_context_tokens` and `context_usage_ratio`; nothing surfaces them yet,
  and `graph.max_turns` still counts turns rather than money.
- **Observability.** OpenTelemetry spans stitched across BFF → kernel → your
  backend, structured logs, and a pluggable error sink. Debugging a bad answer
  currently means reading server logs by hand.
- **Browser-level tests.** The resume path is covered thoroughly on the server
  and not at all in a browser; the front-end has no test suite yet.
- **Conformance in CI.** Run the contract suite against the examples on every
  push, so the contract cannot drift from its reference implementations.

### 3 · Scale past one process

- Move the in-flight run registry off process memory (a shared event log plus
  pub/sub) so resuming works behind more than one worker
- Graceful shutdown that drains turns in flight instead of dropping them
- Rate limiting at the BFF

*Why here:* the single-worker assumption is the main thing between this and a
deployment that can be restarted without losing replies mid-generation.

### 4 · Meet the ecosystem where it is

[MCP](https://modelcontextprotocol.io) has become the common way to expose tools
to models — it moved to vendor-neutral governance under the Linux Foundation in
late 2025 and is supported across every major model provider. CogriaAgent's
`CatalogProvider` and `ActionExecutor` seams already have the right shape for it.

- **Consume MCP.** An MCP-backed catalog provider and executor, so a backend that
  already speaks MCP needs no CogriaAgent-specific endpoints at all
- **Expose MCP.** Serve your action catalog as an MCP server, so the actions you
  write here are reusable by other clients
- **Keep the safety layer on top.** MCP has no human-approval primitive. Wrapping
  MCP tools in propose/confirm — a summary a person reads, a one-shot token bound
  to the parameters — is precisely what CogriaAgent adds over calling them
  directly, and it should apply to an MCP tool exactly as it does to a native one

*Why not sooner:* interoperability is worth more once the thing is installable and
trustworthy. Done earlier it would just widen the surface with nobody using it.

### 5 · Make the agent measurably better

Past roughly twenty actions, tool selection quality becomes the product.

- Prompt versioning that does not require a restart
- An eval harness: fixtures, scoring, and regression gates in CI, so a prompt
  change that degrades tool choice fails the build instead of shipping
- Progressive disclosure for large catalogs — `SKILL.md`-style documents
  describing when to use a group of actions, injected only when relevant, so a
  hundred actions do not bloat every prompt. Execution still goes through the
  catalog schema
- Diagnostics for why a given action was or was not chosen

### 6 · Widen the surface

- **A headless UI package.** Split `agent-ui` into a library (chat runtime,
  adapter, artifact registry, components) plus a thin app shell, so it can be
  embedded in an app that is not this Next.js one — and so downstream products
  stop having to fork the UI to change it
- A Node backend SDK alongside `backend-py`
- A renderer plugin API, and more builtin artifact renderers
- A documentation site

## Not on the roadmap

Saying no is part of a roadmap. CogriaAgent is deliberately not:

- **A general LLM orchestration library.** LangChain and LangGraph do that well
  and this is built on them.
- **A RAG framework.** Attachments inject documents into a turn; retrieval over a
  corpus belongs in an action you write.
- **Multi-tenant.** No tenant dimension anywhere, by design. Run one configured
  instance per tenant.
- **A model abstraction layer.** Any OpenAI-compatible endpoint works; anything
  more is the gateway's job.

## Known gaps

Worth knowing before you build on this:

- **Estimated tokens are estimated.** Compaction thresholds are decided from a
  local estimate (`tiktoken` when it can load its tables, a character heuristic
  otherwise), not from a provider's own count. It is close enough to decide when
  to compact and is not a billing figure.
- **`context_window` is configuration, not discovery.** The kernel does not ask
  the provider how large the window is, so a model swapped without updating
  `summarizer.context_window` keeps the old threshold.
- **No migrations.** The SQL backend creates its schema through `create_all()`,
  which is a development convenience. A production deployment owns its own
  migrations.
- **Single-process assumptions.** Some in-flight state is per-process, so the
  kernel currently expects a single worker. Multi-worker deployments need a
  shared store first.
- **Pre-1.0 interfaces.** The HTTP contract is the most stable surface; Python
  and TypeScript APIs may still shift.

## Design principles

These do not change:

1. **The business backend never talks to an LLM.** No key, no prompt, no
   LangChain. It answers HTTP.
2. **Business logic is injected, never imported.** The kernel depends on
   protocols, so swapping the domain costs zero kernel changes.
3. **Writes need a human.** Any mutation is proposed, summarised in plain
   language, and executed only against a one-shot params-bound token.
4. **Single-tenant.** No tenant dimension anywhere; run one instance per tenant.
5. **Off by default.** Optional capabilities cost nothing when unused, and
   enabling one never changes existing behaviour.
