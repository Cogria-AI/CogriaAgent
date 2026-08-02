"""Catalog providers. A static (in-process dict) one for examples/tests, and an
HTTP one for business backends that expose GET .../agent-actions/_catalog.

Routing is always an explicit URL the caller supplies — no host or path
heuristics, no tenant dimension.
"""

from __future__ import annotations

import time
from typing import Any

import httpx


class StaticCatalogProvider:
    """Serves a fixed catalog dict. Implements protocols.CatalogProvider."""

    def __init__(self, catalog: dict[str, Any]) -> None:
        self._catalog = catalog

    async def get_catalog(self) -> dict[str, Any]:
        return self._catalog


class HttpCatalogProvider:
    """Fetches the catalog from an explicit URL, with a small TTL cache.

    `bearer` is forwarded for backends that gate the catalog endpoint behind auth.
    """

    def __init__(self, url: str, *, bearer: str = "", ttl_seconds: int = 300) -> None:
        self._url = url
        self._bearer = bearer
        self._ttl = ttl_seconds
        self._payload: dict[str, Any] | None = None
        self._fetched_at = 0.0

    async def get_catalog(self) -> dict[str, Any]:
        now = time.monotonic()
        if self._payload is not None and (now - self._fetched_at) < self._ttl:
            return self._payload
        headers = {"Accept": "application/json"}
        if self._bearer:
            headers["Authorization"] = self._bearer
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=3.0)) as client:
            resp = await client.get(self._url, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"catalog fetch failed: {resp.status_code} {resp.text[:200]}")
        self._payload = resp.json()
        self._fetched_at = now
        return self._payload
