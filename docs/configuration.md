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

### `summarizer`

Long conversations are folded into a running summary so the context window and
cost stay bounded.

| Field | Default | Notes |
|---|---|---|
| `enabled` | `true` | |
| `token_threshold` | `16000` | Summarise once history exceeds this |
| `message_threshold` | `50` | …or this many messages |
| `keep_recent` | `20` | Messages kept verbatim after the summary |
| `prompt` | domain-neutral | Override to preserve domain specifics (*"keep product names, prices"*) |

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
