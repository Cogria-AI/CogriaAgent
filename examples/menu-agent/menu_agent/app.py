"""menu-agent — wire the restaurant actions into the CogriaAgent kernel (mode A).

Compare with examples/todo-agent/todo_agent/app_sdk.py: the wiring is byte-for-byte
the same shape — Registry + RegistryExecutor + RegistryCatalogProvider into
build_app — only the actions and the system prompt differ. That is the proof that
swapping the business domain costs zero kernel changes.

Run:
    OPENAI_API_KEY=... OPENAI_BASE_URL=... AGENT_MODEL=... JWT_SECRET=... \
        uv run uvicorn menu_agent.app:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import os

from cogria_agent import AgentConfig, InMemoryConversationBackend, build_app
from cogria_agent.config import LocaleConfig
from cogria_backend import (
    LoggingAuditSink,
    ProposalTokenService,
    Registry,
    RegistryCatalogProvider,
    RegistryExecutor,
)

from .actions import all_actions
from .dev_exchange import add_dev_exchange


def make_config() -> AgentConfig:
    cfg = AgentConfig.from_env()
    cfg.app_name = "menu-agent-agentserv"
    cfg.locales = LocaleConfig(default="en", names={"en": "English", "zh-CN": "简体中文"})
    cfg.system_prompt = (
        "You are a friendly restaurant menu assistant. Reply in {locale_name} throughout.\n"
        "- Use list_dishes to review the menu; create_dish, update_price and "
        "set_availability to change it.\n"
        "- The write actions need confirmation: the system shows the user a proposal "
        "card — do not restate it, just wait.\n"
        "- When you see `[CONFIRMED]`, re-call the same write tool with the proposal_token. "
        "When you see `[CANCELLED]`, briefly acknowledge.\n"
        "- Never invent dishes, ids or prices; if unsure, call list_dishes first."
    )
    return cfg


_config = make_config()
_registry = Registry(all_actions())
_tokens = ProposalTokenService(os.environ.get("JWT_SECRET") or "dev-proposal-secret")
_executor = RegistryExecutor(_registry, _tokens, audit=LoggingAuditSink())

app = build_app(
    _config,
    conversation_backend=InMemoryConversationBackend(),
    action_executor=_executor,
    catalog_provider=RegistryCatalogProvider(_registry),
)

add_dev_exchange(app, _config)
