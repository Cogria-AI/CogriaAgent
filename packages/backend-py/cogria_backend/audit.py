"""Audit sink — write actions record an append-only trail (audit-log.md). The
sink is pluggable; the framework default just logs. A project wires its own
(DB table, etc.). The action orchestrator calls write() with a structured record."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("cogria.audit")


@runtime_checkable
class AuditSink(Protocol):
    async def write(self, record: dict[str, Any]) -> None: ...


class NullAuditSink:
    async def write(self, record: dict[str, Any]) -> None:  # noqa: D401
        return None


class LoggingAuditSink:
    async def write(self, record: dict[str, Any]) -> None:
        logger.info(
            "audit action=%s status=%s target=%s by=%s",
            record.get("action_name"),
            record.get("status"),
            record.get("target_id"),
            record.get("user_id"),
        )
