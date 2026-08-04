"""The injection seams that keep the kernel business-agnostic.

The orchestration kernel (graph / tools / server / summarizer) never talks to a
concrete backend, LLM provider, prompt, or catalog directly — it depends only on
these Protocols. An embedding project (or example) supplies implementations.

This is the line between "framework" and "business":
- ConversationBackend  — where conversations/messages are persisted.
- ActionExecutor       — how a business action actually runs (in-process vs HTTP).
- CatalogProvider      — where the tool catalog comes from (static dict vs HTTP).
- SystemPromptProvider — the role/behaviour prompt (per locale).
- LLMFactory           — which chat/summary models to use.

Attachments (optional; only needed when uploads are enabled) add three more:
- AttachmentStore      — where the bytes live (local disk / S3 / …).
- AttachmentRepository — where the metadata + extracted text live.
- DocumentExtractor    — how a PDF/Word/sheet becomes text the LLM can read.

No tenant dimension anywhere (single-tenant/no-tenant by design). Attachment
ownership rides on the JWT `sub` instead.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# Public conversation ids are opaque strings (the SQL backend mints UUIDs). They
# are what appears in URLs and on the wire; a backend's internal key is its own
# business.
ConversationId = str

# Longest stored/derived conversation title. Shared with the front-end, which
# truncates to the same length so a rename round-trips unchanged.
TITLE_MAX_CHARS = 60


def derive_title(first_message_text: str | None) -> str:
    """A conversation's fallback title: its opening message, collapsed and
    elided. Used when the user has never renamed it, which is most of them."""
    collapsed = " ".join((first_message_text or "").split())
    if not collapsed:
        return ""
    if len(collapsed) <= TITLE_MAX_CHARS:
        return collapsed
    return collapsed[:TITLE_MAX_CHARS].rstrip() + "…"


@runtime_checkable
class ConversationBackend(Protocol):
    """Persistence seam. The kernel persists at the turn boundary and replays
    history for context. Default impl is in-memory (see inmemory.py); a project
    may swap in SQLite/Postgres or its own HTTP backend."""

    async def create_conversation(
        self,
        *,
        first_message: str,
        model: str | None,
        user_id: str | None = None,
        first_content: dict[str, Any] | None = None,
    ) -> ConversationId:
        """Start a conversation and persist its first user message.

        `first_content` is the full content dict when the message carries more
        than text (e.g. {"text": …, "attachments": […]}); when omitted the
        backend stores {"text": first_message}. `first_message` remains the
        plain-text form — backends use it for titles/search.

        `user_id` is the JWT `sub` of the owner. Store it: every user-facing
        read is filtered by it, and without it any authenticated caller could
        read and continue anyone else's conversation.
        """
        ...

    async def list_conversations(
        self, *, user_id: str | None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """That user's conversations, newest first, for a history sidebar.

        Returns dicts of {id, title, message_count, created_at}. Must filter by
        `user_id` and exclude soft-deleted rows — this is a user-facing read.
        """
        ...

    async def soft_delete_conversation(self, conversation_id: ConversationId) -> bool:
        """Hide a conversation from its owner's history without destroying it.
        Idempotent: re-deleting keeps the original timestamp."""
        ...

    async def fetch_history(self, conversation_id: ConversationId, *, for_llm: bool = True) -> list[dict[str, Any]]: ...

    async def append_messages(
        self,
        conversation_id: ConversationId,
        *,
        messages: list[dict[str, Any]],
        usage: dict[str, int] | None,
        model: str | None,
    ) -> dict[str, Any]: ...

    async def fetch_meta(self, conversation_id: ConversationId) -> dict[str, Any]: ...

    async def fetch_messages_full(self, conversation_id: ConversationId) -> list[dict[str, Any]]: ...

    async def save_summary(self, conversation_id: ConversationId, *, summary: str, through_index: int) -> None:
        """Fold messages [0:through_index] into `summary`. The backend decides how
        it's stored and how fetch_history(for_llm=True) replays it (summary head +
        unsummarized tail)."""
        ...


@runtime_checkable
class ActionExecutor(Protocol):
    """Execution seam. Given a tool name + args, run the business action and
    return the standard envelope (contract). `query` carries propose/confirm
    flags (e.g. {"dry_run": "1"}). `context` carries per-request identity the
    LLM never sees — conventionally {"claims": <jwt claims>, "bearer": <token>};
    in-process impls read claims for the AgentContext, HTTP impls forward bearer.
    In-process impls call services directly; HTTP impls forward to a business
    backend's /agent-actions/{slug}."""

    async def invoke(
        self,
        *,
        name: str,
        args: dict[str, Any],
        query: dict[str, str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


@runtime_checkable
class CatalogProvider(Protocol):
    """Catalog seam. Returns {"actions": [...]} describing the available action
    tools. Static (dict) or HTTP-backed; caching is the impl's concern."""

    async def get_catalog(self) -> dict[str, Any]: ...


@runtime_checkable
class AttachmentStore(Protocol):
    """Blob seam. Bytes only — metadata lives in the AttachmentRepository.

    `storage_key` is derived server-side from the content digest and is opaque
    to callers; an implementation must never build it from a client-supplied
    filename (path traversal). The default impl writes to a local directory;
    swapping in S3/GCS means implementing these three methods.
    """

    async def put(self, *, data: bytes, sha256: str, mime: str) -> str: ...

    async def get(self, storage_key: str) -> bytes: ...

    async def delete(self, storage_key: str) -> None: ...


@runtime_checkable
class AttachmentRepository(Protocol):
    """Attachment metadata seam — the same durability story as conversations.

    Every read takes `owner_sub` (the uploader's JWT `sub`) and must filter on
    it: there is no code path in the kernel that fetches an attachment without
    proving ownership.
    """

    async def create(self, **fields: Any) -> dict[str, Any]: ...

    async def mark_ready(
        self,
        attachment_id: str,
        *,
        kind: str,
        page_count: int | None,
        extracted_text: str | None,
    ) -> None: ...

    async def mark_failed(self, attachment_id: str, *, error: str) -> None: ...

    async def get(self, attachment_id: str, *, owner_sub: str) -> dict[str, Any] | None: ...

    async def get_many(self, ids: list[str], *, owner_sub: str) -> list[dict[str, Any]]: ...

    async def find_by_sha256(self, sha256: str, *, owner_sub: str) -> dict[str, Any] | None: ...

    async def count_by_storage_key(self, storage_key: str) -> int:
        """How many rows still point at this blob.

        Storage is content-addressed, so identical bytes from two users (or a
        re-upload after a failed extraction) share one blob. Deleting an
        attachment may only delete the bytes once this returns 0 — otherwise one
        user's delete would empty another's attachment.
        """
        ...

    async def delete(self, attachment_id: str, *, owner_sub: str) -> dict[str, Any] | None: ...


@runtime_checkable
class DocumentExtractor(Protocol):
    """Extraction seam — bytes to text the model can read.

    Returns {"kind": "image"|"text"|"unsupported", "text": str|None,
             "page_count": int|None, "error": str|None}. `kind="image"` means
    "don't extract, send it to a vision model as-is". Swap in OCR/Docling/a
    cloud service by implementing this one method.
    """

    async def extract(self, *, data: bytes, mime: str, filename: str) -> dict[str, Any]: ...


@runtime_checkable
class SystemPromptProvider(Protocol):
    """Prompt seam. Returns the system prompt for a locale, and the human name
    of the reply language — the agent's persona is a project's to define."""

    def system_prompt(self, *, locale: str | None) -> str: ...

    def locale_name(self, *, locale: str | None) -> str: ...
