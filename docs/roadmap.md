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

**Persistence and attachments.** SQLite/Postgres conversation storage with
automatic history summarisation, plus uploads for images, PDF, Word, Excel and
PowerPoint with server-side extraction, a token budget, and the security
controls listed in [Attachments](./attachments.md).

## In progress

**Resumable streams.** The kernel already survives a client disconnect: the
model run is decoupled from the SSE response, so navigating away mid-reply no
longer kills generation or persists a truncated message, and the turn is written
before `done` is emitted. What remains is the reattach path — conversation
history endpoints, a per-conversation registry of in-flight runs with event
replay, and the front-end reconnect — so returning to a conversation shows a
reply that is still being written.

## Planned

**Cost governance.** Layered interception (rate limit at the BFF, pre-flight
budget check in the kernel, turn count), token accounting written back per turn,
usage aggregation, a quota endpoint contract, and a usage banner in the UI. Today
`graph.max_turns` is the only runaway-cost guard.

**Operational governance.** Prompt versioning that does not require a restart, an
eval harness with fixtures and scoring, full error-state coverage with reconnect,
pluggable observability hooks, and a staged-rollout switch.

**Progressive disclosure for tool descriptions.** Optional `SKILL.md`-style
documents describing when to use a group of actions, injected only when relevant,
so large catalogs do not bloat every prompt. Execution would still go through the
catalog schema.

**Release engineering.** Published packages, CI running lint, tests and build,
semantic versioning, and a documentation site.

## Known gaps

Worth knowing before you build on this:

- **No conversation history API.** The kernel persists conversations but exposes
  no endpoint to list or fetch them, and the UI has no conversation route — a
  refresh mid-conversation loses the thread on screen (the data is safe). This is
  the missing half of resumable streams.
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
