"""Context-window summarization, fire-and-forget after a turn is persisted.

When a conversation grows past the configured thresholds, fold its older
messages into a running summary so replays stay within budget. Business-neutral:
the summary instruction comes from config (a project may add domain hints).
Failures are swallowed — summarization is best-effort and must never break chat.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .config import SummarizerConfig
from .protocols import ConversationBackend

logger = logging.getLogger("cogria.summarizer")


def _render(messages: list[dict[str, Any]]) -> str:
    """Flatten persisted rows into a plain transcript for the summary LLM."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content") or {}
        if isinstance(content, dict):
            text = content.get("text") or ""
            if not text and content.get("result") is not None:
                text = json.dumps(content["result"], ensure_ascii=False)[:500]
            # Once a turn is folded into the summary its attachments are no
            # longer replayed, so the file names have to survive here or the
            # assistant forgets a document was ever shared.
            names = [
                a.get("name")
                for a in (content.get("attachments") or [])
                if isinstance(a, dict) and a.get("name")
            ]
            if names:
                text = f"{text} [attached files: {', '.join(names)}]".strip()
        else:
            text = str(content)
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


async def maybe_summarize(
    *,
    backend: ConversationBackend,
    llm_factory: Any,
    config: SummarizerConfig,
    conversation_id: Any,
) -> bool:
    """Summarize if over threshold. Returns True if a summary was written."""
    if not config.enabled:
        return False
    try:
        meta = await backend.fetch_meta(conversation_id)
        msg_count = int(meta.get("message_count", 0) or 0)
        tokens = int(meta.get("total_input_tokens", 0) or 0) + int(meta.get("total_output_tokens", 0) or 0)
        if msg_count <= config.message_threshold and tokens <= config.token_threshold:
            return False

        full = await backend.fetch_messages_full(conversation_id)
        cutoff = len(full) - config.keep_recent
        if cutoff <= 0:
            return False

        transcript = _render(full[:cutoff])
        if not transcript.strip():
            return False

        llm = llm_factory.summary_llm()
        resp = await llm.ainvoke(
            [SystemMessage(content=config.prompt), HumanMessage(content=transcript)]
        )
        summary = resp.content if isinstance(resp.content, str) else str(resp.content)

        await backend.save_summary(conversation_id, summary=summary.strip(), through_index=cutoff)
        logger.info("summarized conv=%s through=%d", conversation_id, cutoff)
        return True
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning("summarize failed conv=%s: %s", conversation_id, e)
        return False
