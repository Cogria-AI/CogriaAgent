"""Propose/confirm tokens.

token = base64url(HMAC-SHA256(subject|action|fingerprint|iat, secret))[:32]
The full params + fingerprint are stashed under a key with a TTL; confirm
verifies subject/action/fingerprint, then atomically one-shot-consumes the token
so an LLM can't replay it (or replay it with different params). Single-tenant:
bound to subject (user), no tenant.

The store is pluggable: InMemoryProposalStore for dev/tests, a Redis-backed one
for production (same atomic `add` semantics).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Protocol, runtime_checkable

TTL_SECONDS = 300
KEY_PREFIX = "agent:proposal:"


@runtime_checkable
class ProposalStore(Protocol):
    async def put(self, key: str, value: dict[str, Any], ttl: int) -> None: ...
    async def get(self, key: str) -> dict[str, Any] | None: ...
    async def add(self, key: str, ttl: int) -> bool:
        """SET-if-absent marker. Returns True iff this caller set it (atomic)."""
        ...
    async def forget(self, key: str) -> None: ...


class InMemoryProposalStore:
    """Process-local store with TTL. Not for multi-process production (use Redis)."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, dict[str, Any]]] = {}
        self._markers: dict[str, float] = {}

    def _expired(self, exp: float) -> bool:
        return exp < time.time()

    async def put(self, key: str, value: dict[str, Any], ttl: int) -> None:
        self._data[key] = (time.time() + ttl, value)

    async def get(self, key: str) -> dict[str, Any] | None:
        item = self._data.get(key)
        if not item:
            return None
        exp, value = item
        if self._expired(exp):
            self._data.pop(key, None)
            return None
        return value

    async def add(self, key: str, ttl: int) -> bool:
        existing = self._markers.get(key)
        if existing is not None and not self._expired(existing):
            return False
        self._markers[key] = time.time() + ttl
        return True

    async def forget(self, key: str) -> None:
        self._data.pop(key, None)


def _fingerprint(params: dict[str, Any]) -> str:
    """Canonical fingerprint: drop control fields, recursively key-sort, sha256."""
    clean = {k: v for k, v in params.items() if k not in ("proposal_token", "_confirmed")}
    canonical = json.dumps(clean, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ProposalTokenService:
    def __init__(self, secret: str, store: ProposalStore | None = None, *, ttl: int = TTL_SECONDS) -> None:
        if not secret:
            raise ValueError("ProposalTokenService requires a non-empty secret")
        self._secret = secret.encode("utf-8")
        self._store = store or InMemoryProposalStore()
        self._ttl = ttl

    async def issue(self, *, subject: str, action: str, params: dict[str, Any]) -> str:
        iat = int(time.time())
        fp = _fingerprint(params)
        raw = f"{subject}|{action}|{fp}|{iat}".encode()
        mac = hmac.new(self._secret, raw, hashlib.sha256).digest()
        token = base64.urlsafe_b64encode(mac).decode().rstrip("=")[:32]
        await self._store.put(
            KEY_PREFIX + token,
            {"subject": subject, "action": action, "fingerprint": fp, "params": params, "iat": iat},
            self._ttl,
        )
        return token

    async def consume(self, token: str, *, subject: str, action: str, params: dict[str, Any]) -> dict[str, Any] | None:
        payload = await self._store.get(KEY_PREFIX + token)
        if not isinstance(payload, dict):
            return None
        if (
            payload.get("subject") != subject
            or payload.get("action") != action
            or payload.get("fingerprint") != _fingerprint(params)
        ):
            return None
        # Atomic one-shot: the loser of a concurrent double-confirm gets None.
        if not await self._store.add(KEY_PREFIX + "used:" + token, self._ttl):
            return None
        await self._store.forget(KEY_PREFIX + token)
        return payload

    async def discard(self, token: str) -> None:
        await self._store.forget(KEY_PREFIX + token)
