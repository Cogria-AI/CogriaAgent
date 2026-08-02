# Getting started

From nothing to a working conversational agent. Budget about ten minutes.

## Prerequisites

| | Why |
|---|---|
| [uv](https://docs.astral.sh/uv/) | Runs the Python services without manual virtualenv work |
| Node 20+ and pnpm | Builds and serves the chat UI |
| Redis | Caches the exchanged JWT on the BFF side |
| An OpenAI-compatible endpoint | Any gateway works: OpenAI, OpenRouter, DeepSeek, a self-hosted proxy |

The model must support tool calling. `gpt-4o-mini`, `deepseek-chat` and most
current instruction-tuned models do. Image uploads additionally need a
vision-capable model — see [Attachments](./attachments.md).

## Path A — scaffold your own agent

```bash
node packages/create-cogria-agent/index.js my-agent
# after publication: npx create-cogria-agent my-agent
```

You get:

```
my-agent/
├── agent/                  # your backend: actions + kernel wiring (Python)
│   └── app/
│       ├── actions.py      # ← your business actions live here
│       ├── store.py        # demo in-memory store (replace with your database)
│       ├── app.py          # wires actions into the kernel
│       └── dev_exchange.py # DEV-only session → JWT stub
├── web/                    # front-end wiring
├── docker-compose.yml      # agentserv + Redis (+ optional Postgres)
└── .env.example
```

Configure and run it:

```bash
cd my-agent/agent
cp .env.example .env          # set OPENAI_BASE_URL, OPENAI_API_KEY,
                              #     AGENT_MODEL, JWT_SECRET
uv run --with pytest --with pytest-asyncio pytest tests
uv run uvicorn app.app:app --host 127.0.0.1 --port 8001
```

The scaffold ships a working task-list agent: add a task, change a priority,
view the list. Replace those three actions in `agent/app/actions.py` with your
own domain — the kernel needs no changes.

## Path B — run an example first

If you would rather see it working before writing anything:

```bash
cd examples/todo-agent
export OPENAI_BASE_URL=https://your-gateway/v1
export OPENAI_API_KEY=sk-...
export AGENT_MODEL=deepseek-chat
export JWT_SECRET=$(openssl rand -base64 48)
uv run uvicorn todo_agent.app:app --host 127.0.0.1 --port 8001
```

[`examples/menu-agent`](../examples/menu-agent) is the same picture in a
write-heavy domain, declared through the backend SDK — a good next read once the
todo one works.

> Both examples mount a **dev-only** `/agent-auth/exchange` that mints a JWT
> unconditionally. It exists so you can run the stack without building auth
> first. Never deploy it.

## Start the front-end

Whichever path you took, the UI is the same:

```bash
cd packages/agent-ui
cp .env.example .env.local
pnpm install
pnpm dev
```

Set these in `.env.local`:

```bash
AGENTSERV_URL=http://127.0.0.1:8001
EXCHANGE_ENDPOINT=http://127.0.0.1:8001/agent-auth/exchange
SUPPORTED_LOCALES=en,zh-CN
```

Open <http://localhost:3003/en>.

## Your first conversation

Three things to try, each exercising a different part of the machinery:

| Say | What happens |
|---|---|
| *"What's on my todo list?"* | A read action runs immediately — no confirmation needed |
| *"Add a todo: buy milk"* | A write action proposes, a confirm card appears, approval executes it |
| *"Show my todos as a table"* | The builtin `render_report` artifact tool opens the side panel |

If the model answers in prose but never calls a tool, the usual cause is a vague
action `description`. It is the only thing the model has to go on — say what the
action does *and when to use it*.

## Writing your first action

With the Python SDK, an action is a class. A read:

```python
from cogria_backend import AgentAction, AgentContext

class ListInvoices(AgentAction):
    name = "list_invoices"
    description = (
        "List invoices, optionally filtered by status. Use when the user asks "
        "what invoices exist, or about unpaid or overdue bills."
    )

    def params_schema(self):
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["paid", "unpaid", "overdue"]},
            },
            "required": [],
            "additionalProperties": False,
        }

    async def run(self, params, ctx: AgentContext):
        return {"invoices": await invoices.query(status=params.get("status"))}

    def read_message(self, data, params):
        return f"{len(data['invoices'])} invoice(s) found."
```

A write differs in three ways: set `requires_confirm = True`, implement
`handle()` instead of `run()`, and supply `proposal_summary()` — the sentence
the user reads on the confirm card before approving.

```python
class VoidInvoice(AgentAction):
    name = "void_invoice"
    description = "Void an invoice. Requires confirmation. Use when the user wants to cancel a bill."
    requires_confirm = True

    def params_schema(self):
        return {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
        }

    async def handle(self, params, ctx):        # runs only after approval
        return await invoices.void(params["id"])

    def proposal_summary(self, params):
        return f"Void invoice #{params['id']}"

    def success_message(self, result, params):
        return f"Invoice #{result['id']} has been voided."
```

The SDK handles the rest: the dry run, the token, the audit entry, and the
envelope.

### Writing descriptions that work

The `description` field is prompt engineering, and it is the highest-leverage
text in your integration.

- Say what it does **and when to use it**: *"Use when the user asks about…"*
- Name the vocabulary your users actually use, including synonyms.
- State preconditions the model should respect: *"Requires an existing customer."*
- Keep parameter meanings in the schema's per-property `description`, not in prose.

## Going further

- Persistence and file uploads: [Configuration](./configuration.md) ·
  [Attachments](./attachments.md)
- A non-Python backend: [Backend contract](../packages/contract/CONTRACT.md), then
  run the conformance suite against it
- How the pieces fit: [Architecture](./architecture.md)
