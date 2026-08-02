"""CogriaAgent HTTP contract — shared pydantic types.

The single source of truth for the wire shapes a business backend must speak:
the action envelope, the catalog, propose/confirm, and exchange. agentserv and
backend-py both import these so the contract can't drift between them.

No tenant dimension (single-tenant by design).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorObj(BaseModel):
    """The `error` member of a failure envelope. `message`/`hint` are LLM-facing
    natural language in the user's locale — never technical jargon."""

    code: str
    message: str
    hint: str | None = None


class Envelope(BaseModel):
    """Unified action response envelope (action-api.md §1.3).

    success: { ok: true,  data, message, audit_id? }
    failure: { ok: false, error, audit_id? }
    propose: { ok: true,  data: { requires_confirm, proposal_token, summary }, message }
    """

    ok: bool
    data: dict[str, Any] | None = None
    message: str | None = None
    error: ErrorObj | None = None
    audit_id: str | None = None


class ProposeData(BaseModel):
    """The `data` of a propose (dry_run) response — what drives the ConfirmCard."""

    requires_confirm: bool = True
    proposal_token: str
    summary: str = ""


class CatalogAction(BaseModel):
    """One action as advertised to the LLM (action-api.md §1.7).

    `name` is snake_case (authoritative, LLM-facing). `url_slug` is kebab-case,
    mechanically `name.replace('_','-')`.
    """

    name: str
    url_slug: str
    description: str
    params_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    returns_schema: dict[str, Any] | None = None
    requires_confirm: bool = False
    ability: str | None = None


class Catalog(BaseModel):
    """GET {base}/agent-actions/_catalog response."""

    actions: list[CatalogAction] = Field(default_factory=list)


class ExchangeResponse(BaseModel):
    """POST {base}/agent-auth/exchange response — mints the short-lived JWT the
    BFF forwards to agentserv."""

    token: str
    expires_at: int
    user: dict[str, Any] | None = None


# Canonical error codes the contract defines for the propose/confirm flow. A
# backend may add its own domain codes; these are the ones agentserv/the
# conformance suite key on.
PROPOSAL_REQUIRED = "PROPOSAL_REQUIRED"
PROPOSAL_EXPIRED = "PROPOSAL_EXPIRED"
VALIDATION_FAILED = "VALIDATION_FAILED"
FORBIDDEN = "FORBIDDEN"


def url_slug_for(name: str) -> str:
    """The mechanical name -> url_slug mapping (snake_case -> kebab-case)."""
    return name.replace("_", "-")
