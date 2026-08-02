"""SqlConversationBackend — the durable default persistence.

Same `ConversationBackend` protocol as `InMemoryConversationBackend`, but rows
survive a restart. One implementation covers SQLite (dev, zero setup) and
Postgres (prod) via SQLAlchemy Core; see sqlschema.py for the tables.

    pip install "cogria-agentserv[sql]"            # SQLite
    pip install "cogria-agentserv[postgres]"       # + asyncpg

    backend = SqlConversationBackend("sqlite+aiosqlite:///./var/agent.db")
    await backend.create_all()   # dev convenience; use migrations in prod

Message rows come back in exactly the dict shape the in-memory backend returns,
so graph.history_to_messages / summarizer work against either unchanged.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .sqlschema import attachments as attachments_table
from .sqlschema import conversations, enable_sqlite_foreign_keys, message_attachments, metadata
from .sqlschema import messages as messages_table

_TITLE_MAX = 80


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _row_to_message(row: Any) -> dict[str, Any]:
    """Persisted row -> the dict shape the kernel replays (see inmemory.py)."""
    out: dict[str, Any] = {
        "id": row.id,
        "role": row.role,
        "content": row.content or {},
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
    if row.parent_index is not None:
        out["parent_index"] = row.parent_index
    if row.input_tokens is not None:
        out["input_tokens"] = row.input_tokens
    if row.output_tokens is not None:
        out["output_tokens"] = row.output_tokens
    if row.finish_reason:
        out["finish_reason"] = row.finish_reason
    if row.error:
        out["error"] = row.error
    return out


class SqlConversationBackend:
    """Implements protocols.ConversationBackend on SQLAlchemy Core."""

    def __init__(
        self,
        url: str | None = None,
        *,
        engine: AsyncEngine | None = None,
        echo: bool = False,
    ) -> None:
        if engine is None:
            if not url:
                raise ValueError("SqlConversationBackend needs a url or an engine")
            engine = create_async_engine(url, echo=echo, future=True)
        self._engine = engine
        enable_sqlite_foreign_keys(engine)
        # Appends compute the next `seq` from MAX(seq). A single-process
        # agentserv serializes them here; the (conversation_id, seq) unique
        # constraint is the backstop if you ever run several writers.
        self._write_lock = asyncio.Lock()

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    async def create_all(self) -> None:
        """Create the schema if absent. Fine for dev/SQLite; prod should run
        real migrations instead."""
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def dispose(self) -> None:
        await self._engine.dispose()

    async def create_conversation(
        self,
        *,
        first_message: str,
        model: str | None,
        first_content: dict[str, Any] | None = None,
    ) -> int:
        now = _now()
        content = first_content or {"text": first_message}
        async with self._write_lock, self._engine.begin() as conn:
            result = await conn.execute(
                insert(conversations).values(
                    title=(first_message or "").strip()[:_TITLE_MAX] or None,
                    model=model,
                    summarized_count=0,
                    total_input_tokens=0,
                    total_output_tokens=0,
                    created_at=now,
                    updated_at=now,
                )
            )
            cid = int(result.inserted_primary_key[0])
            msg = await conn.execute(
                insert(messages_table).values(
                    conversation_id=cid,
                    seq=0,
                    role="user",
                    content=content,
                    created_at=now,
                )
            )
            await self._link_attachments(conn, int(msg.inserted_primary_key[0]), cid, content)
            return cid

    async def fetch_history(
        self, conversation_id: int | str, *, for_llm: bool = True
    ) -> list[dict[str, Any]]:
        cid = int(conversation_id)
        async with self._engine.connect() as conn:
            conv = (
                await conn.execute(select(conversations).where(conversations.c.id == cid))
            ).one_or_none()
            if conv is None:
                return []
            stmt = select(messages_table).where(messages_table.c.conversation_id == cid)
            if for_llm and conv.summary:
                # Everything up to summarized_count is folded into the summary head.
                stmt = stmt.where(messages_table.c.seq >= conv.summarized_count)
            rows = (await conn.execute(stmt.order_by(messages_table.c.seq))).all()

        out = [_row_to_message(r) for r in rows]
        if for_llm and conv.summary:
            return [{"role": "system", "content": {"text": conv.summary}}, *out]
        return out

    async def append_messages(
        self,
        conversation_id: int | str,
        *,
        messages: list[dict[str, Any]],  # noqa: A002 — protocol keyword
        usage: dict[str, int] | None,
        model: str | None,
    ) -> dict[str, Any]:
        cid = int(conversation_id)
        now = _now()
        async with self._write_lock, self._engine.begin() as conn:
            exists = await conn.scalar(select(conversations.c.id).where(conversations.c.id == cid))
            if exists is None:
                raise KeyError(f"no conversation {cid}")

            max_seq = await conn.scalar(
                select(func.coalesce(func.max(messages_table.c.seq), -1)).where(
                    messages_table.c.conversation_id == cid
                )
            )
            next_seq = (int(max_seq) if max_seq is not None else -1) + 1

            # Row-at-a-time (a turn is 1-4 rows): we need each message id to
            # write the message_attachments links.
            for offset, m in enumerate(messages):
                content = m.get("content") or {}
                res = await conn.execute(
                    insert(messages_table).values(
                        conversation_id=cid,
                        seq=next_seq + offset,
                        role=m.get("role", "user"),
                        content=content,
                        parent_index=m.get("parent_index"),
                        input_tokens=m.get("input_tokens"),
                        output_tokens=m.get("output_tokens"),
                        finish_reason=m.get("finish_reason"),
                        error=m.get("error"),
                        created_at=now,
                    )
                )
                await self._link_attachments(conn, int(res.inserted_primary_key[0]), cid, content)

            values: dict[str, Any] = {"updated_at": now}
            if model:
                values["model"] = model
            if usage:
                values["total_input_tokens"] = conversations.c.total_input_tokens + int(
                    usage.get("input_tokens", 0) or 0
                )
                values["total_output_tokens"] = conversations.c.total_output_tokens + int(
                    usage.get("output_tokens", 0) or 0
                )
            await conn.execute(
                update(conversations).where(conversations.c.id == cid).values(**values)
            )

            total = await conn.scalar(
                select(func.count())
                .select_from(messages_table)
                .where(messages_table.c.conversation_id == cid)
            )
        return {"ok": True, "message_count": int(total or 0)}

    async def fetch_meta(self, conversation_id: int | str) -> dict[str, Any]:
        cid = int(conversation_id)
        async with self._engine.connect() as conn:
            conv = (
                await conn.execute(select(conversations).where(conversations.c.id == cid))
            ).one_or_none()
            if conv is None:
                return {}
            count = await conn.scalar(
                select(func.count())
                .select_from(messages_table)
                .where(messages_table.c.conversation_id == cid)
            )
        return {
            "id": conv.id,
            "title": conv.title,
            "model": conv.model,
            "message_count": int(count or 0),
            "total_input_tokens": conv.total_input_tokens,
            "total_output_tokens": conv.total_output_tokens,
            "summarized_count": conv.summarized_count,
        }

    async def fetch_messages_full(self, conversation_id: int | str) -> list[dict[str, Any]]:
        cid = int(conversation_id)
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(messages_table)
                    .where(messages_table.c.conversation_id == cid)
                    .order_by(messages_table.c.seq)
                )
            ).all()
        return [_row_to_message(r) for r in rows]

    async def save_summary(
        self, conversation_id: int | str, *, summary: str, through_index: int
    ) -> None:
        cid = int(conversation_id)
        async with self._engine.begin() as conn:
            total = int(
                await conn.scalar(
                    select(func.count())
                    .select_from(messages_table)
                    .where(messages_table.c.conversation_id == cid)
                )
                or 0
            )
            await conn.execute(
                update(conversations)
                .where(conversations.c.id == cid)
                .values(
                    summary=summary,
                    summarized_count=max(0, min(through_index, total)),
                    updated_at=_now(),
                )
            )

    async def delete_conversation(self, conversation_id: int | str) -> None:
        """Not part of the protocol, but the natural counterpart of create —
        used by retention/cleanup. Messages cascade; attachment rows survive
        (their own retention decides), with conversation_id set to NULL."""
        async with self._engine.begin() as conn:
            await conn.execute(
                delete(conversations).where(conversations.c.id == int(conversation_id))
            )

    async def _link_attachments(
        self, conn: Any, message_id: int, conversation_id: int, content: dict[str, Any]
    ) -> None:
        """Mirror content["attachments"] into the link table and stamp the
        attachment rows with the conversation they were sent in.

        The JSON copy inside the message is what replay/rendering read (no join
        needed); this table is what gives referential integrity and answers
        "which conversations used this file".
        """
        refs = content.get("attachments") if isinstance(content, dict) else None
        if not refs:
            return
        ids = [r.get("id") for r in refs if isinstance(r, dict) and r.get("id")]
        if not ids:
            return
        known = set(
            (
                await conn.execute(
                    select(attachments_table.c.id).where(attachments_table.c.id.in_(ids))
                )
            )
            .scalars()
            .all()
        )
        rows = [
            {"message_id": message_id, "attachment_id": aid, "ordinal": i}
            for i, aid in enumerate(ids)
            if aid in known
        ]
        if not rows:
            return
        await conn.execute(insert(message_attachments), rows)
        await conn.execute(
            update(attachments_table)
            .where(
                attachments_table.c.id.in_([r["attachment_id"] for r in rows]),
                attachments_table.c.conversation_id.is_(None),
            )
            .values(conversation_id=conversation_id)
        )
