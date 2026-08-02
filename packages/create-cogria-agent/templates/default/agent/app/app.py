"""Wire your actions into the CogriaAgent kernel. You rarely edit this file —
the interesting code is in actions.py.

Run:
    uv run uvicorn app.app:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import os

from cogria_agent import AgentConfig, build_app
from cogria_agent.config import LocaleConfig
from cogria_agent.wiring import seams_from_env
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
    # Reads OPENAI_* / JWT_SECRET / AGENT_MODEL from env, then layers your prompt
    # and locales on top.
    cfg = AgentConfig.from_env()
    cfg.app_name = "__PROJECT_NAME__-agentserv"
    cfg.locales = LocaleConfig(default="en", names={"en": "English", "zh-CN": "简体中文"})
    cfg.system_prompt = (
        "You are a friendly task assistant. Reply in {locale_name} throughout.\n"
        "- Use list_tasks to see tasks, add_task to create, set_priority to reprioritize.\n"
        "- add_task and set_priority are write actions: the system asks the user to "
        "confirm first — do not restate the proposal, just wait.\n"
        "- When you see `[CONFIRMED]`, re-call the same write tool with the proposal_token. "
        "When you see `[CANCELLED]`, briefly acknowledge.\n"
        "- Never invent tasks or ids; if unsure, call list_tasks first."
    )
    return cfg


_config = make_config()
_registry = Registry(all_actions())
_tokens = ProposalTokenService(os.environ.get("JWT_SECRET") or "dev-proposal-secret")
_executor = RegistryExecutor(_registry, _tokens, audit=LoggingAuditSink())

# Persistence + file uploads come from env (see env.example): with nothing set
# this runs fully in-memory and uploads are off. Set AGENT_DB_URL to keep
# conversations across restarts, AGENT_ATTACHMENTS_DIR to accept files.
_seams = seams_from_env()
_seams.prepare()

app = build_app(
    _config,
    action_executor=_executor,
    catalog_provider=RegistryCatalogProvider(_registry),
    **_seams.build_kwargs(),
)

# DEV ONLY — a real backend mints the JWT from the user's session. Remove this
# and point EXCHANGE_ENDPOINT at your real /agent-auth/exchange for production.
add_dev_exchange(app, _config)
