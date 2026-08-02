"""InMemoryAttachmentRepository — the zero-config metadata store.

Mirrors InMemoryConversationBackend: fine for dev/tests/examples, gone on
restart. Pair it with SqlAttachmentRepository (sqlrepo.py) for anything real —
attachments are supposed to outlive the process, same as chat history.

Ownership is enforced here, not by the caller: every read takes `owner_sub` and
a record belonging to somebody else is indistinguishable from a missing one.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

# Fields safe to hand back to the browser: no storage_key (internal), no
# owner_sub (identity), no extracted_text (can be megabytes; the LLM reads it
# server-side, the client never needs it).
PUBLIC_FIELDS = (
    "id",
    "filename",
    "mime",
    "size_bytes",
    "kind",
    "status",
    "page_count",
    "extracted_chars",
    "error",
    "created_at",
)


def public_view(record: dict[str, Any]) -> dict[str, Any]:
    return {k: record.get(k) for k in PUBLIC_FIELDS}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class InMemoryAttachmentRepository:
    """Implements protocols.AttachmentRepository."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create(self, **fields: Any) -> dict[str, Any]:
        record = {
            "conversation_id": None,
            "page_count": None,
            "extracted_text": None,
            "extracted_chars": None,
            "error": None,
            "expires_at": None,
            "created_at": _now_iso(),
            **fields,
        }
        async with self._lock:
            self._rows[record["id"]] = record
        return dict(record)

    async def mark_ready(
        self,
        attachment_id: str,
        *,
        kind: str,
        page_count: int | None,
        extracted_text: str | None,
    ) -> None:
        async with self._lock:
            row = self._rows.get(attachment_id)
            if row is None:
                return
            row.update(
                kind=kind,
                status="ready",
                page_count=page_count,
                extracted_text=extracted_text,
                extracted_chars=len(extracted_text) if extracted_text else 0,
            )

    async def mark_failed(self, attachment_id: str, *, error: str) -> None:
        async with self._lock:
            row = self._rows.get(attachment_id)
            if row is not None:
                row.update(status="failed", error=error)

    async def get(self, attachment_id: str, *, owner_sub: str) -> dict[str, Any] | None:
        row = self._rows.get(attachment_id)
        if row is None or row.get("owner_sub") != owner_sub:
            return None
        return dict(row)

    async def get_many(self, ids: list[str], *, owner_sub: str) -> list[dict[str, Any]]:
        out = []
        for aid in ids:
            row = self._rows.get(aid)
            if row is not None and row.get("owner_sub") == owner_sub:
                out.append(dict(row))
        return out

    async def find_by_sha256(self, sha256: str, *, owner_sub: str) -> dict[str, Any] | None:
        # Only a finished extraction is worth reusing — handing back a row that
        # is still extracting (or failed) would dedup away the text itself.
        for row in self._rows.values():
            if (
                row.get("sha256") == sha256
                and row.get("owner_sub") == owner_sub
                and row.get("status") == "ready"
            ):
                return dict(row)
        return None

    async def count_by_storage_key(self, storage_key: str) -> int:
        return sum(1 for row in self._rows.values() if row.get("storage_key") == storage_key)

    async def delete(self, attachment_id: str, *, owner_sub: str) -> dict[str, Any] | None:
        async with self._lock:
            row = self._rows.get(attachment_id)
            if row is None or row.get("owner_sub") != owner_sub:
                return None
            return self._rows.pop(attachment_id)
