"""Todo agent — SDK variant. Same kernel, but actions come from backend-py's
Registry via RegistryExecutor + RegistryCatalogProvider (mode A, embedded). The
agentserv kernel is untouched; only the executor/catalog seams change.

Run:
    OPENAI_API_KEY=... OPENAI_BASE_URL=... AGENT_MODEL=... JWT_SECRET=... \
        uv run uvicorn todo_agent.app_sdk:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import os

from cogria_agent import InMemoryConversationBackend, build_app
from cogria_backend import (
    LoggingAuditSink,
    ProposalTokenService,
    Registry,
    RegistryCatalogProvider,
    RegistryExecutor,
)

from .app import make_config
from .dev_exchange import add_dev_exchange
from .sdk_actions import AddTodo, CompleteTodo, ListTodos

_config = make_config()
_config.app_name = "todo-agentserv-sdk"

_registry = Registry([ListTodos(), AddTodo(), CompleteTodo()])
_tokens = ProposalTokenService(os.environ.get("JWT_SECRET") or "dev-proposal-secret")
_executor = RegistryExecutor(_registry, _tokens, audit=LoggingAuditSink())

app = build_app(
    _config,
    conversation_backend=InMemoryConversationBackend(),
    action_executor=_executor,
    catalog_provider=RegistryCatalogProvider(_registry),
)

add_dev_exchange(app, _config)
