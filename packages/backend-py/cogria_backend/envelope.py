"""Envelope builders — the wire shapes from the contract (action-api.md §1.3)."""

from __future__ import annotations

from typing import Any


def ok(data: dict[str, Any] | None = None, message: str | None = None, *, audit_id: str | None = None) -> dict[str, Any]:
    env: dict[str, Any] = {"ok": True, "data": data or {}}
    if message is not None:
        env["message"] = message
    if audit_id is not None:
        env["audit_id"] = audit_id
    return env


def fail(code: str, message: str, *, hint: str | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if hint is not None:
        error["hint"] = hint
    return {"ok": False, "error": error}


def propose(token: str, summary: str) -> dict[str, Any]:
    return {
        "ok": True,
        "data": {"requires_confirm": True, "proposal_token": token, "summary": summary},
        "message": summary,
    }
