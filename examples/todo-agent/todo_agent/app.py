"""Todo agent example — wire the in-memory backends into the CogriaAgent kernel.

This is the entire integration: pick the seams, call build_app. No business
logic touches the kernel; the kernel carries no todo knowledge.

Run:
    OPENAI_API_KEY=... OPENAI_BASE_URL=... AGENT_MODEL=... JWT_SECRET=... \
        uv run uvicorn todo_agent.app:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

from cogria_agent import (
    AgentConfig,
    StaticCatalogProvider,
    build_app,
)
from cogria_agent.config import AuthConfig, LLMConfig, LocaleConfig
from cogria_agent.wiring import seams_from_env

from .actions import CATALOG, TodoExecutor


def make_config() -> AgentConfig:
    # Reads OPENAI_* / JWT_SECRET / AGENT_MODEL from env, then layers a
    # todo-flavoured system prompt + locales on top. Nothing here is in the kernel.
    cfg = AgentConfig.from_env()
    cfg.app_name = "todo-agentserv"
    cfg.locales = LocaleConfig(default="en", names={"en": "English", "zh-CN": "简体中文"})
    cfg.system_prompt = (
        "You are a friendly todo assistant. Reply in {locale_name} throughout.\n"
        "- Use list_todos to see tasks, add_todo to create, complete_todo to finish.\n"
        "- add_todo and complete_todo are write actions: the system will ask the user "
        "to confirm first — do not restate the proposal, just wait.\n"
        "- When you see `[CONFIRMED]`, re-call the same write tool with the proposal_token. "
        "When you see `[CANCELLED]`, briefly acknowledge.\n"
        "- Never invent tasks or ids; if unsure, call list_todos first."
    )
    return cfg


# Module-level singletons so the in-memory store survives across requests.
_executor = TodoExecutor()
_config = make_config()

# Persistence + attachments come from env: nothing set = in-memory, no uploads.
#   AGENT_DB_URL=sqlite+aiosqlite:///./var/agent.db
#   AGENT_ATTACHMENTS_DIR=./var/attachments   (+ AGENT_VISION_MODEL for images)
_seams = seams_from_env()
_seams.prepare()

app = build_app(
    _config,
    action_executor=_executor,
    catalog_provider=StaticCatalogProvider(CATALOG),
    **_seams.build_kwargs(),
)


# DEV ONLY: stand in for the business backend's session→JWT exchange.
from .dev_exchange import add_dev_exchange  # noqa: E402

add_dev_exchange(app, _config)
