"""Shared SQLAlchemy Core schema for the built-in durable persistence.

One MetaData for everything agentserv owns: conversations + messages (the
`ConversationBackend` seam) and attachments + message_attachments (the
`AttachmentRepository` seam). Core tables, not ORM models — the kernel does a
handful of well-known statements, so a mapper layer would be weight without
payoff, and Core keeps SQLite (dev) and Postgres (prod) on one code path.

Requires the `sql` extra: pip install "cogria-agentserv[sql]"

No tenant dimension (single-tenant by design). Attachment ownership is carried
by `attachments.owner_sub` — the JWT `sub` of whoever uploaded the file.
"""

from __future__ import annotations

import uuid

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.ext.asyncio import AsyncEngine

metadata = MetaData()


def enable_sqlite_foreign_keys(engine: AsyncEngine) -> None:
    """SQLite ignores foreign keys unless each connection opts in.

    Without this, ON DELETE CASCADE silently does nothing on SQLite and does
    fire on Postgres — the same code producing different data depending on the
    dev/prod database. Both backends call this on construction.
    """
    if engine.dialect.name != "sqlite":
        return

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragma(dbapi_connection, _record):  # pragma: no cover - driver callback
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


conversations = Table(
    "conversations",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    # The only id that leaves this process. The serial primary key stays
    # internal so foreign keys and ordering are untouched, while URLs and the
    # API expose an opaque value — a sequential id in a URL both leaks how many
    # conversations exist and invites probing for someone else's.
    Column(
        "public_id",
        String(36),
        nullable=False,
        unique=True,
        index=True,
        default=lambda: str(uuid.uuid4()),
    ),
    # The JWT `sub` that owns this conversation. Nullable because a project may
    # run the kernel without user identity at all; when it is set, every
    # user-facing read filters on it.
    Column("user_id", String(200), index=True),
    Column("title", String(200)),
    # Soft delete: rows are never removed, every user-facing read filters here
    # so history stays available for support and audit.
    Column("deleted_at", DateTime(timezone=True)),
    Column("model", String(200)),
    # Running summary of messages [0:summarized_count] (see summarizer.py).
    Column("summary", Text),
    Column("summarized_count", Integer, nullable=False, default=0, server_default="0"),
    Column("total_input_tokens", Integer, nullable=False, default=0, server_default="0"),
    Column("total_output_tokens", Integer, nullable=False, default=0, server_default="0"),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

messages = Table(
    "messages",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "conversation_id",
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    # Dense 0-based position within the conversation. `summarized_count` and the
    # summarizer's `through_index` are indexes into this same sequence.
    Column("seq", Integer, nullable=False),
    Column("role", String(20), nullable=False),
    # {"text": …} | {"text": …, "tool_calls": […], "attachments": […]}
    #             | {"tool_call_id": …, "name": …, "result": …}
    Column("content", JSON, nullable=False),
    Column("parent_index", Integer),
    Column("input_tokens", Integer),
    Column("output_tokens", Integer),
    Column("finish_reason", String(40)),
    Column("error", JSON),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("conversation_id", "seq", name="uq_messages_conv_seq"),
    Index("ix_messages_conv_seq", "conversation_id", "seq"),
)

attachments = Table(
    "attachments",
    metadata,
    # att_<uuid4hex> — unguessable and non-sequential, so ids can't be walked.
    Column("id", String(40), primary_key=True),
    # JWT `sub` of the uploader. Checked on every read; there is no path that
    # returns an attachment without matching it.
    Column("owner_sub", String(200), nullable=False),
    # Filled in when the attachment is first sent with a message (upload can
    # legitimately precede the conversation's existence).
    Column("conversation_id", Integer, ForeignKey("conversations.id", ondelete="SET NULL")),
    Column("filename", String(300), nullable=False),
    # Sniffed from magic bytes, not from the client's declared type.
    Column("mime", String(150), nullable=False),
    Column("size_bytes", BigInteger, nullable=False),
    Column("sha256", String(64), nullable=False),
    # Derived server-side from the digest — never contains user input.
    Column("storage_key", String(200), nullable=False),
    Column("kind", String(20), nullable=False),  # image | document — what the file IS
    Column("status", String(20), nullable=False),  # extracting | ready | failed
    Column("page_count", Integer),
    Column("extracted_text", Text),
    Column("extracted_chars", Integer),
    Column("error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True)),
    Index("ix_attachments_owner", "owner_sub", "created_at"),
    Index("ix_attachments_sha", "owner_sub", "sha256"),
)

message_attachments = Table(
    "message_attachments",
    metadata,
    Column("message_id", Integer, ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True),
    Column(
        "attachment_id",
        String(40),
        ForeignKey("attachments.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("ordinal", Integer, nullable=False, default=0, server_default="0"),
)
