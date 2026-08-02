"""seams_from_env() — pick persistence + attachment seams from the environment.

Every integration writes the same fifteen lines otherwise: in-memory when
nothing is configured, SQL when AGENT_DB_URL is set, attachments when
AGENT_ATTACHMENTS_DIR is set. Nothing here is required — a project can always
construct the seams itself and pass them to build_app().

    AGENT_DB_URL=sqlite+aiosqlite:///./var/agent.db   # or postgresql+asyncpg://…
    AGENT_ATTACHMENTS_DIR=./var/attachments
    AGENT_VISION_MODEL=gpt-4o-mini                    # needed for image uploads

    seams = seams_from_env()
    seams.prepare()                                   # create tables (dev)
    app = build_app(config, action_executor=…, catalog_provider=…, **seams.build_kwargs())
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from .inmemory import InMemoryConversationBackend


@dataclass
class EnvSeams:
    conversation_backend: Any
    attachment_store: Any | None = None
    attachment_repo: Any | None = None
    durable: bool = False

    def build_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"conversation_backend": self.conversation_backend}
        if self.attachment_store is not None and self.attachment_repo is not None:
            kwargs["attachment_store"] = self.attachment_store
            kwargs["attachment_repo"] = self.attachment_repo
        return kwargs

    def prepare(self) -> None:
        """Create the schema (dev convenience — prod runs migrations instead).

        Safe to call at import time: it runs in a throwaway event loop and then
        disposes the engine, so no connection is left bound to a dead loop when
        uvicorn starts its own.
        """
        if not self.durable:
            return

        async def _run() -> None:
            await self.conversation_backend.create_all()
            await self.conversation_backend.dispose()

        asyncio.run(_run())


def seams_from_env() -> EnvSeams:
    db_url = (os.environ.get("AGENT_DB_URL") or "").strip()
    attachments_dir = (os.environ.get("AGENT_ATTACHMENTS_DIR") or "").strip()

    if db_url:
        from .attachments import SqlAttachmentRepository
        from .sqlbackend import SqlConversationBackend

        backend = SqlConversationBackend(db_url)
        # One engine, one database: a message and its files commit together.
        repo: Any = SqlAttachmentRepository(engine=backend.engine)
        durable = True
    else:
        from .attachments import InMemoryAttachmentRepository

        backend = InMemoryConversationBackend()
        repo = InMemoryAttachmentRepository()
        durable = False

    store = None
    if attachments_dir:
        from .attachments import LocalDiskAttachmentStore

        store = LocalDiskAttachmentStore(attachments_dir)

    return EnvSeams(
        conversation_backend=backend,
        attachment_store=store,
        attachment_repo=repo if store is not None else None,
        durable=durable,
    )
