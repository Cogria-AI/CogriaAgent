# @cogria/agent-ui

The CogriaAgent front-end: a Next.js chat UI (assistant-ui) + artifact panel +
custom SSE adapter + i18n, doubling as the BFF that exchanges the session cookie
for a JWT and forwards chat to the private `agentserv`.

Business-agnostic: no tenant, no domain hardcoding. Everything is env-driven
(see `.env.example`). The three builtin artifact renderers (`render_report`,
`preview_image`, `compare_images`) are generic; projects add their own artifact
tools/renderers on top.

## Run (dev)

```bash
cp .env.example .env.local   # point AGENTSERV_URL / EXCHANGE_ENDPOINT at your stack
pnpm install
pnpm dev                     # http://localhost:3003/<locale>
```

Needs a running `agentserv` (the orchestration kernel) and Redis. See
`examples/todo-agent` for a full local stack.

## Wire format (SSE, from agentserv)

`ready` · `conversation` · `text` · `tool_call` · `tool_result` ·
`confirm_required` · `error` · `done`. Parsed by `src/lib/chat-adapter.ts`.
