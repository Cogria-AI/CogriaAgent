"""Per-request identity passed to actions. Built from the JWT claims agentserv
forwards. Single-tenant: subject (user) + role + locale, no tenant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:
    user_id: str
    role: str | None = None
    locale: str | None = None
    claims: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_claims(cls, claims: dict[str, Any] | None) -> AgentContext:
        claims = claims or {}
        return cls(
            user_id=str(claims.get("sub", "") or ""),
            role=claims.get("role"),
            locale=claims.get("locale"),
            claims=claims,
        )
