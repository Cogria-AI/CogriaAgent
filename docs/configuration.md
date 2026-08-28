# Configuration

Every knob, with its default. Two surfaces: `AgentConfig` on the kernel side and
environment variables on the front-end side.

## Kernel — `AgentConfig`

Construct it directly or call `AgentConfig.from_env()`, then pass it to
`build_app()`. Source:
[`config.py`](../packages/agentserv/cogria_agent/config.py).

### Top level

| Field | Env (`from_env`) | Default | Notes |
|---|---|---|---|
| `app_name` | `AGENT_APP_NAME` | `cogria-agentserv` | Shown in `/health` and OpenAPI |
| `env` | `AGENTSERV_ENV` | `dev` | `dev` enables `/docs` and short cache TTLs |
| `system_prompt` | — | see below | `{locale_name}` is substituted at runtime |

The default system prompt instructs the model to reply in the user's language,
explain results rather than dump JSON, wait for confirmation on writes, and
never fabricate data. Override it for a domain persona, or supply a
`SystemPromptProvider` when the prompt must be computed per request.

### `llm`

| Field | Env | Default |
|---|---|---|
| `base_url` | `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| `api_key` | `OPENAI_API_KEY` | *(empty)* |
| `model` | `AGENT_MODEL` | `gpt-4o-mini` |
| `summary_model` | `AGENT_SUMMARY_MODEL` | falls back to `model` |
| `temperature` | — | `0.2` |
| `summary_temperature` | — | `0.3` |

Any OpenAI-compatible gateway works. The model must support tool calling.

### `graph`

| Field | Default | Notes |
|---|---|---|
| `max_turns` | `10` | Hard cap on model/tool round-trips per request. Real flows rarely exceed 4; this is the runaway-loop guard |
| `max_tool_calls_per_turn` | `8` | Hard cap on distinct calls executed from ONE model response |
| `max_overflow_retries` | `1` | Retries after the provider rejects a request as too long. Each costs a forced compaction plus a fresh request |

### `summarizer`

Once a conversation would fill too much of the context window, its older
messages are folded into a checkpoint and replaced by it.

**Set `context_window` to match your model.** Everything else is expressed as a
share of it, and the default is deliberately small: guessing high is the
dangerous direction, because the threshold then sits above the real window,
compaction never fires, and the conversation runs until the provider rejects it.

| Field | Env | Default | Notes |
|---|---|---|---|
| `enabled` | — | `true` | |
| `context_window` | `AGENT_CONTEXT_WINDOW` | `32000` | The model's context window |
| `threshold_ratio` | — | `0.7` | Compact once the next request is estimated to reach this share of the window |
| `retain_ratio` | — | `0.2` | Share of the window kept verbatim as the recent tail |
| `keep_recent` | — | `4` | Floor on the verbatim tail, in messages |
| `max_summary_input_ratio` | — | `0.6` | Cap on what one summarization call may read |
| `reuse_conversation_prefix` | — | `true` | Replay the conversation's own prompt and tools so the call reuses the provider's prompt cache |
| `prompt` | — | structured, domain-neutral | Override to preserve domain specifics (*"keep product names, prices"*) |
| `token_threshold` | — | unset | Deprecated. Retained so an older config still loads; ignored |
| `message_threshold` | — | `200` | A fuse against pathological message counts, not a trigger |

What gets measured is the size of the messages the *next* request will replay.
Lifetime token usage answers a different question: every turn resends the whole
history, so cumulative usage grows quadratically whether or not the conversation
is anywhere near full.

When uploads are enabled, that measurement includes the file content the request
is about to re-attach, not just the reference stored on the message — a photo
costs around 800 tokens and reads as its filename otherwise. Images are priced
by `attachments.image_history_turns` (only the most recent image-bearing turns
are re-sent as pictures); document text is priced by its injection allowance,
converted at the token density observed in the conversation's own text, because
a character budget is not a token budget and the exchange rate is the language.
A deployment whose `attachments.max_chars_total` is a large share of its
`context_window` will therefore read as fuller than it is; lowering the
injection budget lowers both the estimate and the real cost.

Checkpoints are incremental — each one merges the previous checkpoint with the
span added since — and they replay as a **user** message, because the operating
prompt should be the only system-role instruction the model receives.

Compaction never deletes anything. `summarized_count` is a cursor; every message
stays in the backend, which is what makes [`recall`](#recall) possible.

`reuse_conversation_prefix` is a permission, not a command: replaying the prefix
only pays off against the model that warmed the cache, so pointing
`llm.summary_model` at a different (usually cheaper) model automatically falls
back to sending a flat transcript. Without that fallback the setting would be a
pure loss there — a full structured replay instead of a compact transcript,
billed at list price by a model with no cache for it.

`context_window` is a **budget, not a hardware limit**. On a model with a very
large window, setting it to the technical maximum means compaction never fires
until the request is enormous — and every turn resends the whole prompt, so the
bill arrives long before the window does. Pick what one request should be
allowed to cost.

### `prune`

Bounds on a single tool result as it enters history. A result is written once
and resent on every later turn, so an oversized one is charged repeatedly;
trimming it costs no model call.

| Field | Default | Notes |
|---|---|---|
| `enabled` | `true` | |
| `threshold_chars` | `8192` | Prune a result whose serialized form exceeds this |
| `head_chars` | `4096` | Retained from the start when a result is cut as text |
| `tail_chars` | `1024` | Retained from the end |
| `keep_items` | `20` | Rows retained when the oversized part is a list — the common case |

Only the stored copy is trimmed: the model receives the full result during the
turn that requested it. Propose/confirm envelopes are **never** pruned at any
size, because the `proposal_token` has to come back verbatim. Structured
results stay valid JSON — the envelope's own keys are preserved and the
shortened list is marked with `_truncated`.

`head_chars + tail_chars` must be less than `threshold_chars`; `build_app()`
rejects a configuration that could not shrink anything.

### `recall`

Mounts a `history_search` tool so the model can read back exact values from
messages that a checkpoint only summarises. Off by default — a mounted tool adds
its schema to every request.

| Field | Env | Default | Notes |
|---|---|---|---|
| `enabled` | `AGENT_RECALL` | `false` | Set the variable to `1`/`true` to switch it on |
| `max_results` | — | `5` | Hits returned per search |
| `snippet_chars` | — | `400` | Context returned per hit |

Matching is literal and case-insensitive over the conversation's own rows — no
embeddings, no index, no second store to keep in sync. The conversation id is
bound when the tool is built and is never taken from the model, so reading
another conversation is unreachable rather than merely unauthorized.

### `auth`

| Field | Env | Default |
|---|---|---|
| `jwt_secret` | `JWT_SECRET` | *(empty — every request is rejected)* |
| `jwt_issuer` | `JWT_ISSUER` | `cogria-agent` |
| `required_claims` | — | `["exp", "sub"]` |

HS256, shared with whoever mints the token. No tenant claims — single-tenant by
design.

### `locales`

| Field | Default | Notes |
|---|---|---|
| `default` | `en` | |
| `names` | `{"en": "English"}` | Locale code → language name injected into the prompt |

### `attachments`

Tunes uploads; does not enable them. Uploads switch on when `build_app()`
receives an attachment store and repository. Full discussion:
[Attachments](./attachments.md).

| Field | Env | Default |
|---|---|---|
| `vision_model` | `AGENT_VISION_MODEL` | unset → **image uploads are refused** |
| `max_file_bytes` | — | 20 MiB |
| `max_files_per_turn` | — | `5` |
| `max_chars_per_doc` | — | `20000` |
| `max_chars_total` | — | `40000` |
| `image_history_turns` | — | `1` |
| `allowed_mimes` | — | PNG · JPEG · WebP · GIF · PDF · DOCX · XLSX · PPTX · TXT · Markdown · CSV · JSON |
| `extract_timeout_seconds` | — | `30.0` |
| `max_uncompressed_bytes` | — | 200 MiB (zip-bomb guard) |
| `max_zip_entries` | — | `5000` |

SVG and legacy `.doc` are deliberately excluded: SVG is scriptable, `.doc` has no
reliable extractor.

## Kernel — deployment variables

Read by the wiring helpers rather than `AgentConfig` itself:

| Variable | Effect |
|---|---|
| `AGENT_DB_URL` | Durable history. Unset → in-memory, lost on restart. `sqlite+aiosqlite:///./var/agent.db` or `postgresql+asyncpg://…` |
| `AGENT_ATTACHMENTS_DIR` | Enables uploads, stores blobs at this path |
| `AGENT_CONTEXT_WINDOW` | The model's context window, used for compaction thresholds. Default `32000` |
| `AGENT_RECALL` | `1`/`true` mounts the `history_search` tool |

Install the matching extras: `uv add "cogria-agentserv[sql,attachments]"`.

## Front-end and BFF

From [`.env.example`](../packages/agent-ui/.env.example).

### Required

| Variable | Default | Purpose |
|---|---|---|
| `AGENTSERV_URL` | `http://127.0.0.1:8001` | The private kernel the BFF forwards to |
| `EXCHANGE_ENDPOINT` | *(empty)* | Where the BFF POSTs the session cookie to mint a JWT |
| `REDIS_URL` | `redis://127.0.0.1:6379/2` | BFF JWT cache |
| `JWT_CACHE_TTL_SECONDS` | `1500` | Keep below the token's real lifetime |
| `SUPPORTED_LOCALES` | `en` | Comma-separated; the first is the default |

### Branding

| Variable | Default |
|---|---|
| `NEXT_PUBLIC_APP_TITLE` | `Cogria Agent` |
| `NEXT_PUBLIC_APP_DESCRIPTION` | `A conversational agent for your application.` |

### Attachments

| Variable | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_ATTACHMENTS_ENABLED` | `0` | Whether the composer offers the paperclip. Set to `1` only when the kernel's `/health` reports `"attachments": true` |
| `MAX_ATTACHMENT_BYTES` | `20971520` | BFF-side gate; keep at or below the kernel's `max_file_bytes` |
| `NEXT_PUBLIC_MAX_ATTACHMENT_BYTES` | `20971520` | Same number, checked in the browser when a file is picked |
| `NEXT_PUBLIC_ATTACHMENTS_ACCEPT` | *(empty)* | File-picker filter. Set it to exclude `image/*` when no vision model is configured |

### Optional

| Variable | Purpose |
|---|---|
| `NEXT_PUBLIC_LONG_RUNNING_IMAGE_ACTION` | Name of a slow action returning `data.image_url`; the UI paints a placeholder while it runs |

## Minimal working set

```bash
# kernel
export OPENAI_BASE_URL=https://your-gateway/v1
export OPENAI_API_KEY=sk-...
export AGENT_MODEL=deepseek-chat
export JWT_SECRET=$(openssl rand -base64 48)

# front-end (.env.local)
AGENTSERV_URL=http://127.0.0.1:8001
EXCHANGE_ENDPOINT=http://127.0.0.1:8000/agent-auth/exchange
SUPPORTED_LOCALES=en
```

Everything else has a working default.
