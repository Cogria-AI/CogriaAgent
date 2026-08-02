# todo-agent (example)

A non-restaurant example proving the CogriaAgent kernel carries no business
assumptions: an in-memory todo store wired into `agentserv` via the injection
seams (`ActionExecutor` / `ConversationBackend` / `CatalogProvider`). The whole
integration is `todo_agent/app.py` + `todo_agent/actions.py` — the kernel is
untouched.

## Run the full stack locally

Needs: Python (uv), Node (pnpm), Redis, and an OpenAI-compatible LLM endpoint.

**1. Orchestration kernel (this example) on :8001**

```bash
cd examples/todo-agent
export OPENAI_BASE_URL=https://your-openai-compatible-gateway/v1
export OPENAI_API_KEY=sk-...
export AGENT_MODEL=deepseek-chat            # or any tool-use-capable model
export JWT_SECRET=$(openssl rand -base64 48)
uv run uvicorn todo_agent.app:app --host 127.0.0.1 --port 8001
```

> The example also mounts a **dev-only** `/agent-auth/exchange` that mints a JWT
> unconditionally (a real business backend would mint it from the user's
> session). Never expose that in production.

**2. Front-end + BFF on :3003**

```bash
cd packages/agent-ui
cp .env.example .env.local
# set in .env.local:
#   AGENTSERV_URL=http://127.0.0.1:8001
#   EXCHANGE_ENDPOINT=http://127.0.0.1:8001/agent-auth/exchange
#   SUPPORTED_LOCALES=en,zh-CN
pnpm install
pnpm dev
```

Open <http://localhost:3003/en> and try:

- "What's on my todo list?" → calls `list_todos` (read, runs directly)
- "Add a todo: buy milk" → `add_todo` → inline Confirm card → confirm → executes
- "Show my todos as a table" → builtin `render_report` opens the right-hand panel

## Durable history + file uploads (optional)

Both off by default (in-memory, no paperclip). Add to step 1 to switch them on:

```bash
uv add "cogria-agentserv[sql,attachments]"
export AGENT_DB_URL=sqlite+aiosqlite:///./var/agent.db   # history survives restarts
export AGENT_ATTACHMENTS_DIR=./var/attachments           # enables /attachments
export AGENT_VISION_MODEL=gpt-4o-mini                    # optional; without it
                                                         # image uploads are refused
```

…and `NEXT_PUBLIC_ATTACHMENTS_ENABLED=1` in the front-end `.env.local`. Then attach
a PDF or a .docx and ask about its contents; the file's text is injected into the
prompt, and both the message and the attachment row land in the SQLite file.

## Tests

```bash
uv run --with pytest --with pytest-asyncio pytest examples/todo-agent/tests
```

Real-LLM attachment E2E (upload → answer → replay → verify the database):

```bash
export OPENAI_BASE_URL=… OPENAI_API_KEY=… AGENT_MODEL=deepseek-chat
uv run --extra sql --extra attachments python examples/todo-agent/_e2e_attachments.py
```
