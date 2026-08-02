# CogriaAgent — backend HTTP contract

The language-agnostic contract a business backend implements so `agentserv` can
drive it. Implement it by hand in any language, or use a backend SDK
(`backend-py`). Verify with the conformance suite (`cogria_contract.conformance`).
Single-tenant: no tenant in paths or claims.

## Endpoints

### `GET {base}/agent-actions/_catalog`
Returns the tools the LLM may call.
```json
{ "actions": [
  { "name": "create_thing", "url_slug": "create-thing",
    "description": "…what it does + when to use it (English)…",
    "params_schema": { "type": "object", "properties": {…}, "required": […] },
    "returns_schema": { … },          // optional
    "requires_confirm": true,          // write actions
    "ability": "things.create" }       // optional, your authz gate
]}
```
- `name` snake_case (authoritative, LLM-facing). `url_slug` = `name.replace('_','-')`.

### `POST {base}/agent-actions/{url_slug}`  (and `?dry_run=1`)
Executes an action. Body = flat JSON of `params_schema`. Returns an **envelope**:
```json
// success
{ "ok": true, "data": { … }, "message": "human, in the user's locale", "audit_id": "…?" }
// failure
{ "ok": false, "error": { "code": "…", "message": "…", "hint": "…?" } }
```
`message`/`error.message`/`hint` must be natural language in the user's locale —
the LLM reads them aloud. Never raw validation jargon.

### `POST {base}/agent-auth/exchange`
Mints the short-lived JWT the BFF forwards to agentserv.
```json
{ "token": "eyJ…", "expires_at": 1716001800, "user": { … }? }
```

## Propose / confirm (write actions)

A `requires_confirm` action is two-phase so the user approves side effects:

1. **Propose** — `POST {slug}?dry_run=1` (no token). Validate, do NOT mutate,
   return a proposal:
   ```json
   { "ok": true, "data": { "requires_confirm": true, "proposal_token": "…", "summary": "what will happen" }, "message": "…" }
   ```
2. **Confirm** — `POST {slug}` with `proposal_token` in the body. Execute.

Token rules:
- bound to the caller + action + a **canonical fingerprint of the params**
  (recursively key-sorted JSON, sha256) — a token can't be replayed with
  different params,
- **one-shot** (atomic delete on confirm),
- short TTL (~5 min).

Canonical error codes: `PROPOSAL_REQUIRED` (confirm with no token),
`PROPOSAL_EXPIRED` (unknown/expired/reused/tampered token), `VALIDATION_FAILED`,
`FORBIDDEN`.

## Who talks to the LLM

`agentserv` does — never the business backend. The backend only executes
actions; it holds no LLM key, writes no prompts. agentserv binds auth/identity
server-side (the LLM can't supply it).

## Conformance

```python
from cogria_contract.conformance import run_conformance, format_report
checks = await run_conformance(base, read={...}, write={..., "tampered_args": {...}}, headers={...})
print(format_report(checks))
```
Checks: catalog shape + slug mapping, read envelope, propose returns token,
no-token refused, fingerprint binding, confirm executes, token one-shot.
