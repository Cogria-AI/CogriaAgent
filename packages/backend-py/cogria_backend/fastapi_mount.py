"""FastAPI mount (mode B): expose a Registry over the HTTP contract so a remote
agentserv (with HttpActionExecutor + HttpCatalogProvider) can drive it.

    GET  {base}/agent-actions/_catalog
    POST {base}/agent-actions/{slug}   (?dry_run=1 for propose)

`context_from_request(request) -> claims dict` lets a real backend verify its JWT
and bind identity; the default is anonymous (subject=""), fine for the demo +
conformance suite.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI, Request

from .action import AgentAction
from .audit import AuditSink
from .context import AgentContext
from .envelope import fail
from .proposal import ProposalTokenService
from .registry import Registry


def mount_agent_actions(
    app: FastAPI,
    registry: Registry,
    tokens: ProposalTokenService,
    *,
    audit: AuditSink | None = None,
    context_from_request: Callable[[Request], dict[str, Any] | None] | None = None,
    prefix: str = "",
) -> None:
    base = prefix.rstrip("/")

    @app.get(base + "/agent-actions/_catalog")
    async def _catalog() -> dict[str, Any]:
        return registry.catalog()

    @app.post(base + "/agent-actions/{slug}")
    async def _action(slug: str, request: Request) -> dict[str, Any]:
        action: AgentAction | None = registry.find_by_slug(slug)
        if action is None:
            return fail("UNKNOWN_ACTION", f"No such action: {slug}")
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        if not isinstance(body, dict):
            body = {}
        dry_run = request.query_params.get("dry_run") in ("1", "true", "True")
        token = body.pop("proposal_token", None)
        claims = context_from_request(request) if context_from_request else None
        ctx = AgentContext.from_claims(claims)
        return await action.execute(
            params=body, ctx=ctx, dry_run=dry_run, proposal_token=token, tokens=tokens, audit=audit
        )
