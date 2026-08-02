"""In-process bridge (mode A): expose a Registry to the agentserv kernel as its
ActionExecutor + CatalogProvider, so actions run in the same process — no HTTP
hop, direct access to your services. (Mode B is the FastAPI mount + agentserv's
HttpActionExecutor.)

These duck-type agentserv's protocols.ActionExecutor / protocols.CatalogProvider
without importing agentserv (backend-py stays standalone).
"""

from __future__ import annotations

from typing import Any

from .action import AgentAction
from .audit import AuditSink
from .context import AgentContext
from .envelope import fail
from .proposal import ProposalTokenService
from .registry import Registry


class RegistryExecutor:
    def __init__(self, registry: Registry, tokens: ProposalTokenService, *, audit: AuditSink | None = None) -> None:
        self._registry = registry
        self._tokens = tokens
        self._audit = audit

    async def invoke(
        self,
        *,
        name: str,
        args: dict[str, Any],
        query: dict[str, str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        action: AgentAction | None = self._registry.find_by_name(name)
        if action is None:
            return fail("UNKNOWN_ACTION", f"No such action: {name}")
        ctx = AgentContext.from_claims((context or {}).get("claims"))
        dry_run = bool(query and query.get("dry_run"))
        token = args.get("proposal_token")
        params = {k: v for k, v in args.items() if k != "proposal_token"}
        return await action.execute(
            params=params, ctx=ctx, dry_run=dry_run, proposal_token=token, tokens=self._tokens, audit=self._audit
        )


class RegistryCatalogProvider:
    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    async def get_catalog(self) -> dict[str, Any]:
        return self._registry.catalog()
