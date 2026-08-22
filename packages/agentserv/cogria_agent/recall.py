"""`history_search` — reading back what compaction folded away.

Compaction replaces an older span with a checkpoint, and a checkpoint is lossy
by construction. But the rows themselves are not deleted: `summarized_count` is
a cursor, and every message stays in the backend. This tool is the path back to
them, so "compressed" stops meaning "gone".

It deliberately does not use embeddings. A conversation is a few hundred
messages, the model already knows the exact identifier or phrase it is looking
for, and substring matching over the same rows the backend already stores needs
no index, no extra dependency, and no separate store to keep in sync. That
tradeoff would change for a corpus; it does not for one conversation.

Scoped to the conversation in progress. It reads through `fetch_messages_full`,
which every ConversationBackend already implements, so switching this on
requires no backend change — and cross-conversation reads are not merely
unimplemented but unreachable, since the id is bound at construction and never
taken from the model.

Off by default (`RecallConfig.enabled`): a mounted tool adds its schema to every
request whether or not it is used.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from .config import RecallConfig
from .protocols import ConversationBackend

logger = logging.getLogger("cogria.recall")

#: Injected into the system prompt only when the tool is mounted, so a project
#: with recall switched off keeps the prompt it wrote.
RECALL_GUIDANCE = (
    "\n\nEarlier parts of this conversation may have been condensed into a "
    "checkpoint. When you need the exact wording, a specific number, or a detail "
    "the checkpoint only summarizes, call history_search with a distinctive term "
    "from it rather than guessing or asking the user to repeat themselves."
)


class _HistorySearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        description=(
            "A distinctive word or phrase to look for in this conversation's earlier "
            "messages — an identifier, a name, a number, an error string. Matching is "
            "literal and case-insensitive, so prefer an exact term over a question."
        ),
    )


def _searchable_text(row: dict[str, Any]) -> str:
    """Everything in one persisted row that a search should be able to match."""
    content = row.get("content") or {}
    if not isinstance(content, dict):
        return str(content)

    parts: list[str] = []
    if content.get("text"):
        parts.append(str(content["text"]))
    for call in content.get("tool_calls") or []:
        if isinstance(call, dict):
            parts.append(str(call.get("name") or ""))
            parts.append(json.dumps(call.get("args") or {}, ensure_ascii=False))
    if content.get("result") is not None:
        result = content["result"]
        parts.append(
            result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        )
    for att in content.get("attachments") or []:
        if isinstance(att, dict) and att.get("name"):
            parts.append(str(att["name"]))
    return "\n".join(p for p in parts if p)


def _snippet(text: str, needle: str, width: int) -> str:
    """`width` characters of context around the first match, with ellipses."""
    position = text.lower().find(needle.lower())
    if position < 0:
        return text[:width]
    start = max(0, position - width // 3)
    end = min(len(text), start + width)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def search_rows(
    rows: list[dict[str, Any]], query: str, *, limit: int, snippet_chars: int
) -> list[dict[str, Any]]:
    """Literal, case-insensitive search over persisted rows, newest first.

    Newest first because a repeated identifier is nearly always wanted at its
    most recent mention, and the result set is capped.
    """
    needle = (query or "").strip()
    if not needle:
        return []

    hits: list[dict[str, Any]] = []
    for index in range(len(rows) - 1, -1, -1):
        row = rows[index]
        text = _searchable_text(row)
        if needle.lower() not in text.lower():
            continue
        hits.append(
            {
                "position": index,
                "role": row.get("role", "?"),
                "excerpt": _snippet(text, needle, snippet_chars),
            }
        )
        if len(hits) >= limit:
            break
    return hits


def build_recall_tool(
    *,
    backend: ConversationBackend,
    conversation_id: Any,
    config: RecallConfig,
) -> StructuredTool:
    """One `history_search` tool bound to one conversation.

    The conversation id is closed over rather than accepted as an argument, so
    there is no reachable path by which the model could read another
    conversation — the authorization question never arises because the
    capability is not exposed.
    """

    async def _runner(query: str) -> str:
        try:
            rows = await backend.fetch_messages_full(conversation_id)
        except Exception as e:  # noqa: BLE001 — a search failure must not break chat
            logger.warning("history_search failed conv=%s: %s", conversation_id, e)
            return json.dumps(
                {"ok": False, "error": {"code": "HISTORY_SEARCH_FAILED", "message": str(e)}},
                ensure_ascii=False,
            )

        hits = search_rows(
            rows, query, limit=config.max_results, snippet_chars=config.snippet_chars
        )
        return json.dumps(
            {
                "ok": True,
                "data": {
                    "query": query,
                    "matches": hits,
                    # Named so an empty result reads as "not in this conversation"
                    # rather than as a broken tool.
                    "searched_messages": len(rows),
                },
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        coroutine=_runner,
        name="history_search",
        description=(
            "Search the earlier messages of THIS conversation, including parts that "
            "were condensed into a checkpoint and are no longer shown in full. Use it "
            "to recover an exact value the checkpoint only summarizes. Matching is "
            "literal and case-insensitive."
        ),
        args_schema=_HistorySearchArgs,
    )
