"""In-memory ConversationBackend — the zero-config default for dev/tests/examples.

Business-agnostic. A real deployment swaps in a SQLite/Postgres backend or a
business-backend HTTP client implementing the same ConversationBackend protocol.
State lives in process memory; a restart clears it.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .protocols import derive_title


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _Conversation:
    id: str
    model: str | None
    user_id: str | None = None
    title: str | None = None
    deleted_at: datetime | None = None
    created_at: datetime = field(default_factory=_now)
    messages: list[dict[str, Any]] = field(default_factory=list)
    summary: str | None = None
    summarized_count: int = 0  # how many leading messages are folded into `summary`
    input_tokens: int = 0
    output_tokens: int = 0


class InMemoryConversationBackend:
    """Implements protocols.ConversationBackend."""

    def __init__(self) -> None:
        self._store: dict[str, _Conversation] = {}
        self._lock = asyncio.Lock()

    async def create_conversation(
        self,
        *,
        first_message: str,
        model: str | None,
        user_id: str | None = None,
        first_content: dict[str, Any] | None = None,
    ) -> str:
        async with self._lock:
            cid = str(uuid.uuid4())
            conv = _Conversation(id=cid, model=model, user_id=user_id)
            conv.messages.append(
                {"role": "user", "content": first_content or {"text": first_message}}
            )
            self._store[cid] = conv
            return cid

    async def fetch_history(
        self, conversation_id: str, *, for_llm: bool = True
    ) -> list[dict[str, Any]]:
        conv = self._store.get(str(conversation_id))
        if not conv:
            return []
        if for_llm and conv.summary:
            # A user message, not a system one: the operating prompt is the only
            # system-role instruction the model should get, and a checkpoint in
            # that role reads as one. summarizer.frame_summary() supplies the
            # framing that marks it as background.
            head = [{"role": "user", "content": {"text": conv.summary}}]
            return head + conv.messages[conv.summarized_count :]
        return list(conv.messages)

    async def append_messages(
        self,
        conversation_id: str,
        *,
        messages: list[dict[str, Any]],
        usage: dict[str, int] | None,
        model: str | None,
    ) -> dict[str, Any]:
        conv = self._store.get(str(conversation_id))
        if not conv:
            raise KeyError(f"no conversation {conversation_id}")
        conv.messages.extend(messages)
        if usage:
            conv.input_tokens += int(usage.get("input_tokens", 0) or 0)
            conv.output_tokens += int(usage.get("output_tokens", 0) or 0)
        if model:
            conv.model = model
        return {"ok": True, "message_count": len(conv.messages)}

    async def fetch_meta(self, conversation_id: str) -> dict[str, Any]:
        conv = self._store.get(str(conversation_id))
        if not conv:
            return {}
        # A NULL title means "never renamed" — derive the same fallback the
        # listing uses so both agree on what this conversation is called.
        title = conv.title
        if not title:
            first = conv.messages[0] if conv.messages else {}
            text = (first.get("content") or {}).get("text") if first.get("role") == "user" else ""
            title = derive_title(text)
        return {
            "id": conv.id,
            "user_id": conv.user_id,
            "title": title or None,
            "deleted_at": conv.deleted_at.isoformat() if conv.deleted_at else None,
            "model": conv.model,
            "message_count": len(conv.messages),
            "total_input_tokens": conv.input_tokens,
            "total_output_tokens": conv.output_tokens,
            "summarized_count": conv.summarized_count,
            # The running checkpoint, so a later compaction can merge into it
            # instead of producing a second, overlapping one.
            "summary": conv.summary,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
        }

    async def fetch_messages_full(self, conversation_id: str) -> list[dict[str, Any]]:
        conv = self._store.get(str(conversation_id))
        return list(conv.messages) if conv else []

    async def list_conversations(
        self, *, user_id: str | None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        owned = [
            c
            for c in self._store.values()
            if c.user_id == user_id and c.deleted_at is None
        ]
        # Newest first. Insertion order is creation order, so reversing is the
        # in-memory equivalent of the SQL backend's ORDER BY id DESC.
        owned.reverse()
        out: list[dict[str, Any]] = []
        for conv in owned[offset : offset + limit]:
            first = conv.messages[0] if conv.messages else {}
            text = (first.get("content") or {}).get("text") if first.get("role") == "user" else ""
            out.append(
                {
                    "id": conv.id,
                    "title": conv.title or derive_title(text),
                    "message_count": len(conv.messages),
                    "created_at": conv.created_at.isoformat() if conv.created_at else None,
                }
            )
        return out

    async def soft_delete_conversation(self, conversation_id: str) -> bool:
        conv = self._store.get(str(conversation_id))
        if not conv:
            return False
        if conv.deleted_at is None:  # idempotent: keep the original timestamp
            conv.deleted_at = _now()
        return True

    async def rename_conversation(self, conversation_id: str, *, title: str | None) -> bool:
        conv = self._store.get(str(conversation_id))
        if not conv:
            return False
        conv.title = title
        return True

    async def save_summary(
        self, conversation_id: str, *, summary: str, through_index: int
    ) -> None:
        conv = self._store.get(str(conversation_id))
        if conv:
            conv.summary = summary
            conv.summarized_count = max(0, min(through_index, len(conv.messages)))
