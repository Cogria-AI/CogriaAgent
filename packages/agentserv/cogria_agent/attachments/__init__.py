"""Attachment support: upload files, feed them to the model, keep them in the DB.

Wire it by passing an AttachmentStore + AttachmentRepository to build_app():

    from cogria_agent.attachments import (
        DefaultDocumentExtractor, InMemoryAttachmentRepository, LocalDiskAttachmentStore,
    )

    app = build_app(
        config,
        ...,
        attachment_store=LocalDiskAttachmentStore("./var/attachments"),
        attachment_repo=InMemoryAttachmentRepository(),   # or SqlAttachmentRepository
    )

Without those two arguments the kernel behaves exactly as it did before
attachments existed: no /attachments routes, and `attachment_ids` in a /chat
body is refused.

SqlAttachmentRepository is exported lazily so this package imports without the
`sql` extra.
"""

from __future__ import annotations

from typing import Any

from .extract import DefaultDocumentExtractor
from .inject import ATTACHMENT_GUIDANCE, Budget, build_turn_content, hydrate_history
from .repo import InMemoryAttachmentRepository, public_view
from .routes import build_attachment_router
from .service import AttachmentRejected, AttachmentService
from .store import LocalDiskAttachmentStore


def __getattr__(name: str) -> Any:
    if name == "SqlAttachmentRepository":
        try:
            from .sqlrepo import SqlAttachmentRepository
        except ImportError as e:  # noqa: BLE001 — re-raise with the fix attached
            raise ImportError(
                'SqlAttachmentRepository needs the "sql" extra: '
                f'pip install "cogria-agentserv[sql]" ({e})'
            ) from e
        return SqlAttachmentRepository
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ATTACHMENT_GUIDANCE",
    "AttachmentRejected",
    "AttachmentService",
    "Budget",
    "DefaultDocumentExtractor",
    "InMemoryAttachmentRepository",
    "LocalDiskAttachmentStore",
    "SqlAttachmentRepository",
    "build_attachment_router",
    "build_turn_content",
    "hydrate_history",
    "public_view",
]
