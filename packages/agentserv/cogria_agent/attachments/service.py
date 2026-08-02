"""AttachmentService — the one place upload rules are enforced.

Ties the three attachment seams together (store + repository + extractor) and
owns the policy the routes and /chat both depend on:

  size cap -> magic-byte sniffing -> mime allowlist -> vision policy ->
  dedup by digest -> store bytes -> persist metadata -> extract text

Rejections are raised as AttachmentRejected with a message written for a human
(the UI shows it verbatim), never as a stack trace or validation jargon.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any

from ..config import AttachmentsConfig
from ..protocols import AttachmentRepository, AttachmentStore, DocumentExtractor
from .extract import IMAGE_MIMES, TEXT_MIMES
from .repo import public_view

logger = logging.getLogger("cogria.attachments")

_EXTENSION_MIMES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
}


class AttachmentRejected(Exception):
    """A refusal the user should read: too big, wrong type, not yours."""

    def __init__(self, message: str, *, status_code: int = 415) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AttachmentService:
    def __init__(
        self,
        *,
        store: AttachmentStore,
        repo: AttachmentRepository,
        extractor: DocumentExtractor,
        config: AttachmentsConfig,
    ) -> None:
        self.store = store
        self.repo = repo
        self.extractor = extractor
        self.config = config

    @property
    def vision_enabled(self) -> bool:
        return bool(self.config.vision_model)

    async def upload(
        self, *, data: bytes, filename: str, declared_mime: str, owner_sub: str
    ) -> dict[str, Any]:
        cfg = self.config
        if not data:
            raise AttachmentRejected("That file is empty.", status_code=422)
        if len(data) > cfg.max_file_bytes:
            limit_mb = cfg.max_file_bytes / (1024 * 1024)
            raise AttachmentRejected(
                f"That file is larger than the {limit_mb:.0f} MB limit.", status_code=413
            )

        mime = self._sniff(data, declared_mime, filename)
        if mime not in cfg.allowed_mimes:
            raise AttachmentRejected(f"{filename}: this file type isn't supported.")
        if mime in IMAGE_MIMES and not self.vision_enabled:
            raise AttachmentRejected(
                "Images can't be read by the current model. Ask an administrator to "
                "configure a vision-capable model to enable image uploads."
            )

        digest = hashlib.sha256(data).hexdigest()
        existing = await self.repo.find_by_sha256(digest, owner_sub=owner_sub)
        if existing is not None:
            # Same bytes, same owner: reuse the extraction instead of redoing it.
            return public_view(existing)

        storage_key = await self.store.put(data=data, sha256=digest, mime=mime)
        record = await self.repo.create(
            id=f"att_{uuid.uuid4().hex}",
            owner_sub=owner_sub,
            filename=(filename or "file")[:300],
            mime=mime,
            size_bytes=len(data),
            sha256=digest,
            storage_key=storage_key,
            kind="image" if mime in IMAGE_MIMES else "document",
            status="extracting",
        )

        result = await self.extractor.extract(data=data, mime=mime, filename=filename)
        if result.get("kind") == "unsupported":
            # Stored and readable by the user, just not by the model — the turn
            # will tell the model why rather than pretending the file isn't there.
            await self.repo.mark_failed(record["id"], error=result.get("error") or "Unreadable.")
        else:
            # `kind` records what the file IS (that's what the UI and the
            # injector branch on), not how the extractor happened to handle it.
            await self.repo.mark_ready(
                record["id"],
                kind="image" if mime in IMAGE_MIMES else "document",
                page_count=result.get("page_count"),
                extracted_text=result.get("text"),
            )
        fresh = await self.repo.get(record["id"], owner_sub=owner_sub)
        return public_view(fresh or record)

    async def resolve(self, ids: list[str], *, owner_sub: str) -> list[dict[str, Any]]:
        """Attachment ids from a /chat body -> full records, in the given order.

        Unknown ids and other people's ids are indistinguishable on purpose.
        """
        if not ids:
            return []
        # Same file listed twice is a client slip, not an attack — collapse it
        # rather than failing the turn (and count the caps against real files).
        ids = list(dict.fromkeys(ids))
        if len(ids) > self.config.max_files_per_turn:
            raise AttachmentRejected(
                f"You can attach at most {self.config.max_files_per_turn} files to one message.",
                status_code=422,
            )
        records = await self.repo.get_many(ids, owner_sub=owner_sub)
        if len(records) != len(ids):
            raise AttachmentRejected("An attachment is missing or isn't yours.", status_code=403)
        return records

    async def load_bytes(self, record: dict[str, Any]) -> bytes:
        return await self.store.get(record["storage_key"])

    async def delete(self, attachment_id: str, *, owner_sub: str) -> bool:
        record = await self.repo.delete(attachment_id, owner_sub=owner_sub)
        if record is None:
            return False
        try:
            # Storage is content-addressed: identical bytes uploaded by another
            # user (or re-uploaded after a failed extraction) share this blob.
            # Only the last reference may take it with them.
            if await self.repo.count_by_storage_key(record["storage_key"]) == 0:
                await self.store.delete(record["storage_key"])
        except Exception as e:  # noqa: BLE001 — metadata is gone; a stray blob is harmless
            logger.warning("blob delete failed for %s: %s", attachment_id, e)
        return True

    def _sniff(self, data: bytes, declared_mime: str, filename: str) -> str:
        """Trust magic bytes over the client's Content-Type and the extension.

        Formats with no magic number (txt/csv/json/md) are accepted only if the
        bytes actually decode as UTF-8 — which is what stops `evil.exe` renamed
        to `notes.txt` from getting through.
        """
        try:
            import filetype  # noqa: PLC0415 — lazy: optional `attachments` extra
        except ImportError as e:  # pragma: no cover - depends on install profile
            raise AttachmentRejected(
                'File uploads need the "attachments" extra on the server.', status_code=500
            ) from e

        guess = filetype.guess(data)
        if guess is not None:
            return guess.mime

        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            raise AttachmentRejected(f"{filename}: this file type isn't supported.") from None

        lowered = (filename or "").lower()
        for ext, mime in _EXTENSION_MIMES.items():
            if lowered.endswith(ext):
                return mime
        if declared_mime in TEXT_MIMES:
            return declared_mime
        return "text/plain"
