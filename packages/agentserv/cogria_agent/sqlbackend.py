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
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, delete, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .protocols import derive_title
from .sqlschema import attachments as attachments_table
from .sqlschema import conversations, enable_sqlite_foreign_keys, message_attachments, metadata
from .sqlschema import messages as messages_table


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

    @staticmethod
    async def _internal_id(conn: Any, public_id: str) -> int | None:
        """Public UUID -> serial primary key. Every method takes the public id;
        this is the single place the two representations meet."""
        return await conn.scalar(
            select(conversations.c.id).where(conversations.c.public_id == str(public_id))
        )

    async def create_conversation(
        self,
        *,
        first_message: str,
        model: str | None,
        user_id: str | None = None,
        first_content: dict[str, Any] | None = None,
    ) -> str:
        now = _now()
        content = first_content or {"text": first_message}
        public_id = str(uuid.uuid4())
        async with self._write_lock, self._engine.begin() as conn:
            result = await conn.execute(
                insert(conversations).values(
                    public_id=public_id,
                    user_id=user_id,
                    title=None,  # NULL means "never renamed" — derive from message 0
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
            return public_id

    async def fetch_history(
        self, conversation_id: str, *, for_llm: bool = True
    ) -> list[dict[str, Any]]:
        async with self._engine.connect() as conn:
            conv = (
                await conn.execute(
                    select(conversations).where(conversations.c.public_id == str(conversation_id))
                )
            ).one_or_none()
            if conv is None:
                return []
            cid = int(conv.id)
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
        conversation_id: str,
        *,
        messages: list[dict[str, Any]],  # noqa: A002 — protocol keyword
        usage: dict[str, int] | None,
        model: str | None,
    ) -> dict[str, Any]:
        now = _now()
        async with self._write_lock, self._engine.begin() as conn:
            cid = await self._internal_id(conn, conversation_id)
            if cid is None:
                raise KeyError(f"no conversation {conversation_id}")
            cid = int(cid)

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

    async def fetch_meta(self, conversation_id: str) -> dict[str, Any]:
        """Everything the ownership gate and the detail view need. Returns {}
        when the id is unknown; `deleted_at` is reported rather than filtered so
        the caller decides what a soft-deleted conversation means to it."""
        async with self._engine.connect() as conn:
            conv = (
                await conn.execute(
                    select(conversations).where(conversations.c.public_id == str(conversation_id))
                )
            ).one_or_none()
            if conv is None:
                return {}
            count = await conn.scalar(
                select(func.count())
                .select_from(messages_table)
                .where(messages_table.c.conversation_id == conv.id)
            )
            # A NULL title means "never renamed". Derive the same fallback the
            # listing uses, so both endpoints agree on what this conversation is
            # called — a detail view and a sidebar row disagreeing reads as a bug.
            title = conv.title
            if not title:
                first = await conn.scalar(
                    select(messages_table.c.content).where(
                        and_(
                            messages_table.c.conversation_id == conv.id,
                            messages_table.c.seq == 0,
                        )
                    )
                )
                title = derive_title((first or {}).get("text") if isinstance(first, dict) else "")
        return {
            "id": conv.public_id,
            "user_id": conv.user_id,
            "title": title or None,
            "deleted_at": conv.deleted_at.isoformat() if conv.deleted_at else None,
            "model": conv.model,
            "message_count": int(count or 0),
            "total_input_tokens": conv.total_input_tokens,
            "total_output_tokens": conv.total_output_tokens,
            "summarized_count": conv.summarized_count,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
        }

    async def fetch_messages_full(self, conversation_id: str) -> list[dict[str, Any]]:
        async with self._engine.connect() as conn:
            cid = await self._internal_id(conn, conversation_id)
            if cid is None:
                return []
            rows = (
                await conn.execute(
                    select(messages_table)
                    .where(messages_table.c.conversation_id == cid)
                    .order_by(messages_table.c.seq)
                )
            ).all()
        return [_row_to_message(r) for r in rows]

    async def list_conversations(
        self, *, user_id: str | None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        cc = conversations.c
        mc = messages_table.c
        # Message count and the opening message (seq 0, which stands in for a
        # title) as correlated subqueries, so listing N conversations stays one
        # round trip instead of 2N.
        count_sq = (
            select(func.count())
            .select_from(messages_table)
            .where(mc.conversation_id == cc.id)
            .scalar_subquery()
        )
        first_sq = (
            select(mc.content)
            .where(and_(mc.conversation_id == cc.id, mc.seq == 0))
            .limit(1)
            .scalar_subquery()
        )
        owned = cc.user_id == user_id if user_id is not None else cc.user_id.is_(None)
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(
                        cc.public_id,
                        cc.created_at,
                        cc.title,
                        count_sq.label("n"),
                        first_sq.label("first"),
                    )
                    # Soft-deleted conversations are gone as far as their owner
                    # is concerned; the rows stay for support and audit.
                    .where(and_(owned, cc.deleted_at.is_(None)))
                    .order_by(cc.id.desc())
                    .limit(limit)
                    .offset(offset)
                )
            ).all()
        out: list[dict[str, Any]] = []
        for r in rows:
            first = r.first if isinstance(r.first, dict) else {}
            text = (first.get("text") or "") if isinstance(first, dict) else ""
            out.append(
                {
                    "id": r.public_id,
                    "title": r.title or derive_title(text),
                    "message_count": int(r.n or 0),
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
            )
        return out

    async def soft_delete_conversation(self, conversation_id: str) -> bool:
        async with self._engine.begin() as conn:
            cid = await self._internal_id(conn, conversation_id)
            if cid is None:
                return False
            # Idempotent: re-deleting keeps the original timestamp.
            await conn.execute(
                update(conversations)
                .where(and_(conversations.c.id == cid, conversations.c.deleted_at.is_(None)))
                .values(deleted_at=_now())
            )
            return True

    async def rename_conversation(self, conversation_id: str, *, title: str | None) -> bool:
        async with self._engine.begin() as conn:
            cid = await self._internal_id(conn, conversation_id)
            if cid is None:
                return False
            await conn.execute(
                update(conversations).where(conversations.c.id == cid).values(title=title)
            )
            return True

    async def save_summary(
        self, conversation_id: str, *, summary: str, through_index: int
    ) -> None:
        async with self._engine.begin() as conn:
            cid = await self._internal_id(conn, conversation_id)
            if cid is None:
                return
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

    async def delete_conversation(self, conversation_id: str) -> None:
        """Hard delete, for retention/cleanup — not the user-facing one (that is
        soft_delete_conversation). Messages cascade; attachment rows survive
        (their own retention decides), with conversation_id set to NULL."""
        async with self._engine.begin() as conn:
            await conn.execute(
                delete(conversations).where(conversations.c.public_id == str(conversation_id))
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
