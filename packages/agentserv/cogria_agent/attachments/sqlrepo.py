"""SqlAttachmentRepository — durable attachment metadata (needs the `sql` extra).

Shares one engine/schema with SqlConversationBackend so a message and the files
it carries commit against the same database; sqlbackend._link_attachments is
what writes the message_attachments rows.

    backend = SqlConversationBackend("sqlite+aiosqlite:///./var/agent.db")
    repo = SqlAttachmentRepository(engine=backend.engine)
    await backend.create_all()          # creates both sets of tables
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ..sqlschema import attachments as attachments_table
from ..sqlschema import enable_sqlite_foreign_keys, metadata

_COLUMNS = tuple(c.name for c in attachments_table.columns)


def _row_to_dict(row: Any) -> dict[str, Any]:
    out = {name: getattr(row, name) for name in _COLUMNS}
    for key in ("created_at", "expires_at"):
        value = out.get(key)
        if isinstance(value, datetime):
            out[key] = value.isoformat()
    return out


class SqlAttachmentRepository:
    """Implements protocols.AttachmentRepository."""

    def __init__(
        self, url: str | None = None, *, engine: AsyncEngine | None = None, echo: bool = False
    ) -> None:
        if engine is None:
            if not url:
                raise ValueError("SqlAttachmentRepository needs a url or an engine")
            engine = create_async_engine(url, echo=echo, future=True)
            enable_sqlite_foreign_keys(engine)
        self._engine = engine

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    async def create_all(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def create(self, **fields: Any) -> dict[str, Any]:
        values = {k: v for k, v in fields.items() if k in _COLUMNS}
        values.setdefault("created_at", datetime.now(timezone.utc))
        async with self._engine.begin() as conn:
            await conn.execute(insert(attachments_table).values(**values))
            row = (
                await conn.execute(
                    select(attachments_table).where(attachments_table.c.id == values["id"])
                )
            ).one()
        return _row_to_dict(row)

    async def mark_ready(
        self,
        attachment_id: str,
        *,
        kind: str,
        page_count: int | None,
        extracted_text: str | None,
    ) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                update(attachments_table)
                .where(attachments_table.c.id == attachment_id)
                .values(
                    kind=kind,
                    status="ready",
                    page_count=page_count,
                    extracted_text=extracted_text,
                    extracted_chars=len(extracted_text) if extracted_text else 0,
                )
            )

    async def mark_failed(self, attachment_id: str, *, error: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                update(attachments_table)
                .where(attachments_table.c.id == attachment_id)
                .values(status="failed", error=error)
            )

    async def get(self, attachment_id: str, *, owner_sub: str) -> dict[str, Any] | None:
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    select(attachments_table).where(
                        attachments_table.c.id == attachment_id,
                        attachments_table.c.owner_sub == owner_sub,
                    )
                )
            ).one_or_none()
        return _row_to_dict(row) if row is not None else None

    async def get_many(self, ids: list[str], *, owner_sub: str) -> list[dict[str, Any]]:
        if not ids:
            return []
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(attachments_table).where(
                        attachments_table.c.id.in_(ids),
                        attachments_table.c.owner_sub == owner_sub,
                    )
                )
            ).all()
        by_id = {r.id: _row_to_dict(r) for r in rows}
        # Preserve the caller's order — it's the order the user attached them.
        return [by_id[i] for i in ids if i in by_id]

    async def find_by_sha256(self, sha256: str, *, owner_sub: str) -> dict[str, Any] | None:
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    select(attachments_table)
                    .where(
                        attachments_table.c.sha256 == sha256,
                        attachments_table.c.owner_sub == owner_sub,
                        attachments_table.c.status == "ready",
                    )
                    .order_by(attachments_table.c.created_at.desc())
                    .limit(1)
                )
            ).one_or_none()
        return _row_to_dict(row) if row is not None else None

    async def count_by_storage_key(self, storage_key: str) -> int:
        async with self._engine.connect() as conn:
            total = await conn.scalar(
                select(func.count())
                .select_from(attachments_table)
                .where(attachments_table.c.storage_key == storage_key)
            )
        return int(total or 0)

    async def delete(self, attachment_id: str, *, owner_sub: str) -> dict[str, Any] | None:
        record = await self.get(attachment_id, owner_sub=owner_sub)
        if record is None:
            return None
        async with self._engine.begin() as conn:
            await conn.execute(
                delete(attachments_table).where(attachments_table.c.id == attachment_id)
            )
        return record
