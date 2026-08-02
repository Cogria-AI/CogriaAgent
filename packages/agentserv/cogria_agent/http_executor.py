"""HttpActionExecutor — dispatch actions to a remote business backend that speaks
the HTTP contract (POST {base}/agent-actions/{slug}). Forwards the per-request
bearer from `context` so the backend authorises as the end user.
Implements protocols.ActionExecutor (adapter mode)."""

from __future__ import annotations

from typing import Any

import httpx


class HttpActionExecutor:
    def __init__(self, base_url: str, *, default_bearer: str = "", timeout: float = 120.0) -> None:
        # 120s default: write actions may call slow external APIs (e.g. image gen).
        self._base = base_url.rstrip("/")
        self._default_bearer = default_bearer
        self._timeout = timeout

    async def invoke(
        self,
        *,
        name: str,
        args: dict[str, Any],
        query: dict[str, str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        slug = name.replace("_", "-")
        bearer = (context or {}).get("bearer") or self._default_bearer
        headers = {"Accept": "application/json"}
        if bearer:
            headers["Authorization"] = bearer
        url = f"{self._base}/agent-actions/{slug}"
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout, connect=5.0)) as client:
            resp = await client.post(url, json=args, params=query or None, headers=headers)
        try:
            return resp.json()
        except Exception:  # noqa: BLE001 — translate a non-JSON / error body into an envelope
            return {
                "ok": False,
                "error": {"code": "BACKEND_ERROR", "message": f"backend {resp.status_code}: {resp.text[:200]}"},
            }
