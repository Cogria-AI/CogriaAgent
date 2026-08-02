"""In-memory ConversationBackend — the zero-config default for dev/tests/examples.

Business-agnostic. A real deployment swaps in a SQLite/Postgres backend or a
business-backend HTTP client implementing the same ConversationBackend protocol.
State lives in process memory; a restart clears it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Conversation:
    id: int
    model: str | None
    messages: list[dict[str, Any]] = field(default_factory=list)
    summary: str | None = None
    summarized_count: int = 0  # how many leading messages are folded into `summary`
    input_tokens: int = 0
    output_tokens: int = 0


class InMemoryConversationBackend:
    """Implements protocols.ConversationBackend."""

    def __init__(self) -> None:
        self._store: dict[int, _Conversation] = {}
        self._next_id = 1
        self._lock = asyncio.Lock()

    async def create_conversation(
        self,
        *,
        first_message: str,
        model: str | None,
        first_content: dict[str, Any] | None = None,
    ) -> int:
        async with self._lock:
            cid = self._next_id
            self._next_id += 1
            conv = _Conversation(id=cid, model=model)
            conv.messages.append({"role": "user", "content": first_content or {"text": first_message}})
            self._store[cid] = conv
            return cid

    async def fetch_history(self, conversation_id: int, *, for_llm: bool = True) -> list[dict[str, Any]]:
        conv = self._store.get(int(conversation_id))
        if not conv:
            return []
        if for_llm and conv.summary:
            head = [{"role": "system", "content": {"text": conv.summary}}]
            return head + conv.messages[conv.summarized_count :]
        return list(conv.messages)

    async def append_messages(
        self,
        conversation_id: int,
        *,
        messages: list[dict[str, Any]],
        usage: dict[str, int] | None,
        model: str | None,
    ) -> dict[str, Any]:
        conv = self._store.get(int(conversation_id))
        if not conv:
            raise KeyError(f"no conversation {conversation_id}")
        conv.messages.extend(messages)
        if usage:
            conv.input_tokens += int(usage.get("input_tokens", 0) or 0)
            conv.output_tokens += int(usage.get("output_tokens", 0) or 0)
        if model:
            conv.model = model
        return {"ok": True, "message_count": len(conv.messages)}

    async def fetch_meta(self, conversation_id: int) -> dict[str, Any]:
        conv = self._store.get(int(conversation_id))
        if not conv:
            return {}
        return {
            "id": conv.id,
            "model": conv.model,
            "message_count": len(conv.messages),
            "total_input_tokens": conv.input_tokens,
            "total_output_tokens": conv.output_tokens,
            "summarized_count": conv.summarized_count,
        }

    async def fetch_messages_full(self, conversation_id: int) -> list[dict[str, Any]]:
        conv = self._store.get(int(conversation_id))
        return list(conv.messages) if conv else []

    async def save_summary(self, conversation_id: int, *, summary: str, through_index: int) -> None:
        conv = self._store.get(int(conversation_id))
        if conv:
            conv.summary = summary
            conv.summarized_count = max(0, min(through_index, len(conv.messages)))
