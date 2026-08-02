# __PROJECT_NAME__

A conversational agent scaffolded with **CogriaAgent**. You write the business
actions; the framework brings the LLM orchestration, chat UI, propose/confirm,
auth, and governance.

```
__PROJECT_NAME__/
├── agent/                  # your backend: actions + kernel wiring (Python)
│   └── app/
│       ├── actions.py      # ← your business actions live here
│       ├── store.py        # demo in-memory store (replace with your DB)
│       ├── app.py          # wires actions into the kernel (don't edit much)
│       └── dev_exchange.py # DEV-only session→JWT stub
├── web/                    # front-end (CogriaAgent agent-ui) wiring
├── docker-compose.yml      # agentserv + redis (+ optional postgres)
└── .env.example
```

The sample agent manages a **task list**: add tasks, change a task's priority,
and view the list. Rename/replace the three actions in `agent/app/actions.py`
to model your own domain — the kernel never changes.

---

## Run it (local, with uv)

The verified-today path. Needs [uv](https://docs.astral.sh/uv/), Node + pnpm
(for the UI), Redis, and an OpenAI-compatible LLM endpoint.

**1. The agent backend (orchestration kernel) on :8001**

```bash
cd agent
cp .env.example .env
# edit .env: set OPENAI_BASE_URL / OPENAI_API_KEY / AGENT_MODEL / JWT_SECRET
set -a; source .env; set +a
uv run uvicorn app.app:app --host 127.0.0.1 --port 8001
```

**2. The front-end + BFF on :3003**

The UI is the framework's `@cogria/agent-ui` (a long-running service you point at
your agent). From the CogriaAgent checkout:

```bash
cd <cogria-agent>/packages/agent-ui
cp .env.example .env.local
#   AGENTSERV_URL=http://127.0.0.1:8001
#   EXCHANGE_ENDPOINT=http://127.0.0.1:8001/agent-auth/exchange
#   SUPPORTED_LOCALES=en,zh-CN
pnpm install && pnpm dev
```

`web/.env.local.example` here holds the exact values to copy.

Open <http://localhost:3003/en> and try:

- "What's on my list?" → `list_tasks` (read, runs directly)
- "Add a task: ship the demo" → `add_task` → Confirm card → confirm → executes
- "Make task 1 high priority" → `set_priority` → Confirm → executes
- "Show my tasks as a table" → builtin `render_report` opens the right-hand panel

---

## Run it (Docker)

`docker-compose.yml` brings up the agent kernel + redis with one command:

```bash
cp .env.example .env       # COGRIA_FRAMEWORK_PATH is pre-filled by the scaffolder
docker compose up
# add postgres too:
docker compose --profile postgres up
```

> **Pre-publish note.** CogriaAgent's packages aren't on PyPI yet (that lands in
> C6). Until then, the compose file installs the framework from a local checkout
> mounted via `COGRIA_FRAMEWORK_PATH` (the scaffolder filled it in for you in
> `.env`). Once the packages publish, swap that mount for a plain
> `pip install cogria-agentserv cogria-backend` in `agent/Dockerfile`.

The front-end still runs separately (step 2 above) — compose covers the backend
kernel + redis, matching the framework's deployment model (agentserv is a
private service; the BFF/UI is the only caller).

## Tests

```bash
cd agent && uv run --with pytest --with pytest-asyncio pytest tests
```
