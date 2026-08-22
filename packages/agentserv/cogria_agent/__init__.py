"""CogriaAgent orchestration kernel — business-agnostic conversational agent core.

Public API:
    build_app                          — construct the FastAPI service
    AgentConfig (+ sub-configs)        — declarative configuration
    InMemoryConversationBackend        — zero-config persistence (dev/tests)
    SqlConversationBackend             — durable persistence (needs the `sql` extra)
    StaticCatalogProvider / HttpCatalogProvider
    DefaultLLMFactory / ConfigSystemPromptProvider
    protocols.*                        — the injection seams

The SQL / attachment pieces are exported lazily (PEP 562) so importing the
kernel never requires the optional `sql` / `attachments` dependencies.
"""

from __future__ import annotations

from typing import Any

from .catalog import HttpCatalogProvider, StaticCatalogProvider
from .config import (
    AgentConfig,
    AttachmentsConfig,
    AuthConfig,
    GraphConfig,
    LLMConfig,
    LocaleConfig,
    PruneConfig,
    RecallConfig,
    SummarizerConfig,
)
from .http_executor import HttpActionExecutor
from .inmemory import InMemoryConversationBackend
from .llm import ConfigSystemPromptProvider, DefaultLLMFactory
from .protocols import (
    ActionExecutor,
    CatalogProvider,
    ConversationBackend,
    SystemPromptProvider,
)
from .server import build_app

_LAZY: dict[str, tuple[str, str]] = {
    # name -> (module, extra that provides its dependencies)
    "SqlConversationBackend": (".sqlbackend", "sql"),
}


def __getattr__(name: str) -> Any:
    """Import optional-dependency exports on first use, with an actionable
    message when the extra isn't installed."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, extra = target
    from importlib import import_module

    try:
        return getattr(import_module(module, __name__), name)
    except ImportError as e:  # noqa: BLE001 — re-raise with the fix attached
        raise ImportError(
            f'{name} needs the "{extra}" extra: pip install "cogria-agentserv[{extra}]" ({e})'
        ) from e


__all__ = [
    "build_app",
    "AgentConfig",
    "AttachmentsConfig",
    "SqlConversationBackend",
    "LLMConfig",
    "GraphConfig",
    "SummarizerConfig",
    "PruneConfig",
    "RecallConfig",
    "AuthConfig",
    "LocaleConfig",
    "InMemoryConversationBackend",
    "StaticCatalogProvider",
    "HttpCatalogProvider",
    "HttpActionExecutor",
    "DefaultLLMFactory",
    "ConfigSystemPromptProvider",
    "ActionExecutor",
    "CatalogProvider",
    "ConversationBackend",
    "SystemPromptProvider",
]
