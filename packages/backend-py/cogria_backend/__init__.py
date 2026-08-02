"""CogriaAgent Python backend SDK — framework-agnostic AgentAction core + bridges.

Define actions as AgentAction subclasses, register them, then either:
  - embed (mode A): RegistryExecutor + RegistryCatalogProvider into agentserv, or
  - serve (mode B): mount_agent_actions(app, ...) to expose the HTTP contract.
"""

from __future__ import annotations

from .action import AgentAction
from .audit import AuditSink, LoggingAuditSink, NullAuditSink
from .context import AgentContext
from .envelope import fail, ok, propose
from .executor import RegistryCatalogProvider, RegistryExecutor
from .proposal import InMemoryProposalStore, ProposalStore, ProposalTokenService
from .registry import Registry

__all__ = [
    "AgentAction",
    "AgentContext",
    "Registry",
    "ProposalTokenService",
    "ProposalStore",
    "InMemoryProposalStore",
    "RegistryExecutor",
    "RegistryCatalogProvider",
    "AuditSink",
    "NullAuditSink",
    "LoggingAuditSink",
    "ok",
    "fail",
    "propose",
]


def mount(*args, **kwargs):
    """Lazy proxy to fastapi_mount.mount_agent_actions (keeps FastAPI optional)."""
    from .fastapi_mount import mount_agent_actions

    return mount_agent_actions(*args, **kwargs)
