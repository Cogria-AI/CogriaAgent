# menu-agent (example)

The **reference** example: a restaurant menu — "create a dish / change a price /
mark something sold out / review the menu" — declared as `backend-py` actions on
the CogriaAgent kernel. Write-heavy on purpose, so the propose/confirm and audit
machinery is visible in every interaction.

Its whole reason to exist is comparison. Put it next to
[`examples/todo-agent`](../todo-agent): the wiring in `menu_agent/app.py` is the
same shape as `todo_agent/app_sdk.py` (Registry → RegistryExecutor →
RegistryCatalogProvider → `build_app`). Only the actions and the system prompt
change. **The kernel (`agentserv`/`agent-ui`) is byte-for-byte identical across
the two domains** — that is the "swap the business, don't touch the framework"
claim, made concrete.

## What's here

| File | Role |
|---|---|
| `menu_agent/store.py` | In-memory menu — stands in for your real DB/models |
| `menu_agent/actions.py` | 4 `AgentAction` subclasses: `list_dishes` (read) + `create_dish` / `update_price` / `set_availability` (writes, confirm) |
| `menu_agent/app.py` | Mode-A embedded wiring into the kernel |
| `menu_agent/dev_exchange.py` | DEV-only session→JWT exchange stub |

The writes show off the SDK's propose/confirm + audit: `update_price` captures
before/after state and emits a `price X → Y` diff summary — an audit trail you
get for free from the base class, without writing any bookkeeping code.

## Run the full stack locally

Same as the todo example, swapping the module name. Needs Python (uv), Node
(pnpm), Redis, and an OpenAI-compatible LLM endpoint.

**1. Orchestration kernel on :8001**

```bash
cd examples/menu-agent
export OPENAI_BASE_URL=https://your-openai-compatible-gateway/v1
export OPENAI_API_KEY=sk-...
export AGENT_MODEL=deepseek-chat            # or any tool-use-capable model
export JWT_SECRET=$(openssl rand -base64 48)
uv run uvicorn menu_agent.app:app --host 127.0.0.1 --port 8001
```

> Mounts a **dev-only** `/agent-auth/exchange`. Never expose it in production.

**2. Front-end + BFF on :3003** — identical to the todo example's step 2
([README](../todo-agent/README.md)); set `AGENTSERV_URL` /
`EXCHANGE_ENDPOINT` to the kernel above.

Open <http://localhost:3003/en> and try:

- "What's on the menu?" → `list_dishes` (read, runs directly)
- "Add a dish: Espresso, 3.00, Drinks" → `create_dish` → Confirm card → confirm
- "Change dish 1 to 14.00" → `update_price` → Confirm → executes (audited diff)
- "Mark dish 1 as sold out" → `set_availability`

## Tests

```bash
uv run --with pytest --with pytest-asyncio pytest examples/menu-agent/tests
```

### Real-LLM end-to-end (manual)

`_e2e_manual.py` drives the kernel's `/chat` SSE in-process (no bound port) through
read → propose → confirm against a real model — proving the full path
LLM ↔ kernel ↔ SDK ↔ actions. Point it at any OpenAI-compatible gateway:

```bash
export OPENAI_BASE_URL=https://your-gateway/v1
export OPENAI_API_KEY=sk-...
export AGENT_MODEL=deepseek-chat            # any tool-use-capable model
uv run python examples/menu-agent/_e2e_manual.py
```
