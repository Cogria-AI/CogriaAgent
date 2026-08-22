"""Token estimation for the messages a turn is about to replay.

The summarizer needs one number: how big is the prompt we are about to send.
That is not the same as what the conversation has cost so far — every turn
resends the whole history, so lifetime usage grows quadratically while the
prompt may be tiny. Measuring the replay set directly is the only judgement
that tracks the thing being budgeted.

Two estimators, in preference order:

1. `tiktoken`, when it is importable and its BPE table loads. Exact for the
   OpenAI families and close enough for the OpenAI-compatible ones.
2. A character heuristic, when it isn't. `len/4` is the usual rule of thumb and
   it badly under-counts CJK — a Chinese character is closer to one token than
   to a quarter of one — so the fallback counts scripts separately.

The tiktoken path loads a BPE table over the network on first use. A deployment
with no egress would hang on it, so the failure is caught once, remembered, and
degrades to the heuristic rather than propagating.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

logger = logging.getLogger("cogria.estimate")

# Role framing, message delimiters, and the JSON scaffolding around a tool call:
# small, fixed, and paid once per message.
_MESSAGE_OVERHEAD = 4

# What one image block costs. Real cost varies by resolution and provider (~85
# tokens for a low-detail thumbnail, over 1000 for a full-detail page scan);
# this sits high on purpose, because under-counting images is what lets a
# conversation sail past the threshold without ever triggering compaction.
_IMAGE_TOKENS = 800

# Non-ASCII scripts that are roughly one token per character rather than the
# ~4 characters per token that Latin text averages. CJK ideographs, kana,
# Hangul, and the full-width forms.
_DENSE_RANGES = (
    (0x3000, 0x30FF),
    (0x3130, 0x318F),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xA960, 0xA97F),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
    (0xFF00, 0xFFEF),
    (0x20000, 0x2FA1F),
)

_encoder: Any | None = None
_encoder_tried = False
_encoder_lock = threading.Lock()


def warm_encoder() -> None:
    """Resolve the encoder off the event loop, in the background.

    The first resolution may fetch a BPE table over the network. Doing that
    inside a request would stall the loop for every other connection on the
    process, and it would happen at exactly the wrong moment — the first
    conversation large enough to need compaction. Called from `build_app()`.
    """
    if _encoder_tried:
        return
    threading.Thread(target=_get_encoder, name="cogria-tiktoken-warm", daemon=True).start()


def _get_encoder() -> Any | None:
    """The tiktoken encoder, or None once we know it isn't available.

    Resolved exactly once: a deployment where the BPE table cannot be fetched
    must degrade to the heuristic quietly instead of retrying on every estimate.
    """
    global _encoder, _encoder_tried
    if _encoder_tried:
        return _encoder
    with _encoder_lock:
        if _encoder_tried:
            return _encoder
        return _resolve_encoder()


def _resolve_encoder() -> Any | None:
    global _encoder, _encoder_tried
    try:
        import tiktoken

        # o200k_base covers the current OpenAI families. Model-specific lookup
        # is deliberately skipped: the model name here is often a
        # non-OpenAI one behind a compatible gateway, and the vocabulary
        # difference is far smaller than the error we are trying to avoid.
        _encoder = tiktoken.get_encoding("o200k_base")
    except Exception as e:  # noqa: BLE001 — any failure means "use the heuristic"
        logger.info("tiktoken unavailable (%s); estimating tokens by character class", e)
        _encoder = None
    finally:
        # Set last: another thread must not read a half-resolved state, and a
        # failure must be remembered as firmly as a success.
        _encoder_tried = True
    return _encoder


def _heuristic(text: str) -> int:
    """Characters to tokens, counting dense scripts at ~1 token each."""
    dense = 0
    for ch in text:
        code = ord(ch)
        for low, high in _DENSE_RANGES:
            if low <= code <= high:
                dense += 1
                break
    loose = len(text) - dense
    # Dense characters are counted slightly under 1:1 — most tokenizers merge
    # some common two-character words — while Latin text keeps the /4 rule.
    return int(dense * 0.9) + -(-loose // 4)


def estimate_text(text: str) -> int:
    """Tokens for one string."""
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is None:
        return _heuristic(text)
    try:
        return len(encoder.encode(text, disallowed_special=()))
    except Exception:  # noqa: BLE001 — a pathological input must not break chat
        return _heuristic(text)


def _estimate_blocks(blocks: Any) -> int:
    """Tokens for multimodal content: text blocks priced, images at a flat rate."""
    if not isinstance(blocks, list):
        return 0
    total = 0
    for block in blocks:
        if not isinstance(block, dict):
            total += estimate_text(str(block))
            continue
        if block.get("type") == "image_url":
            total += _IMAGE_TOKENS
        else:
            total += estimate_text(block.get("text") or "")
    return total


def estimate_message(row: dict[str, Any]) -> int:
    """Tokens for one persisted message row, in the shape the backends store.

    Mirrors `graph.history_to_messages`: whatever that function turns into model
    input is what gets priced here. `blocks` (rebuilt attachment content) wins
    over `text` when present, because that is what actually goes on the wire.
    """
    content = row.get("content") or {}
    if not isinstance(content, dict):
        return estimate_text(str(content)) + _MESSAGE_OVERHEAD

    total = _MESSAGE_OVERHEAD

    blocks = content.get("blocks")
    if blocks:
        total += _estimate_blocks(blocks)
    else:
        total += estimate_text(content.get("text") or "")

    for call in content.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        total += estimate_text(str(call.get("name") or ""))
        total += estimate_text(json.dumps(call.get("args") or {}, ensure_ascii=False))
        total += _MESSAGE_OVERHEAD

    result = content.get("result")
    if result is not None:
        total += estimate_text(
            result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        )

    # Attachment metadata is replayed as a short filename list, not as content.
    for att in content.get("attachments") or []:
        if isinstance(att, dict):
            total += estimate_text(str(att.get("name") or "")) + 2

    return total


def estimate_messages(rows: list[dict[str, Any]]) -> int:
    """Tokens for a whole replay set."""
    return sum(estimate_message(row) for row in rows)


def reset_encoder_cache() -> None:
    """Forget the resolved encoder. For tests that exercise both paths."""
    global _encoder, _encoder_tried
    with _encoder_lock:
        _encoder = None
        _encoder_tried = False
