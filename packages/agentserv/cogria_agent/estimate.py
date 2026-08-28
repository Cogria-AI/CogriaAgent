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

from .config import AttachmentsConfig

logger = logging.getLogger("cogria.estimate")

# Role framing, message delimiters, and the JSON scaffolding around a tool call:
# small, fixed, and paid once per message.
_MESSAGE_OVERHEAD = 4

# What one image block costs. Real cost varies by resolution and provider (~85
# tokens for a low-detail thumbnail, over 1000 for a full-detail page scan);
# this sits high on purpose, because under-counting images is what lets a
# conversation sail past the threshold without ever triggering compaction.
_IMAGE_TOKENS = 800

# What one `<attachment id=… name=… type=… note=… />` element costs beyond its
# filename: the tag scaffolding, which every attachment replays whether or not
# its content does.
_ATTACHMENT_REF_TOKENS = 24

# Tokens per character when a conversation offers no text to sample from —
# the usual Latin rule of thumb. See `_observed_density`.
_DEFAULT_DENSITY = 0.25

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

    # Each attachment replays at minimum as one <attachment …/> element naming
    # it. Whether its CONTENT also replays is a whole-request decision, priced
    # by `estimate_messages` — see `_replay_attachment_tokens`.
    for att in content.get("attachments") or []:
        if isinstance(att, dict):
            total += estimate_text(str(att.get("name") or "")) + _ATTACHMENT_REF_TOKENS

    return total


def _replay_attachment_tokens(
    rows: list[dict[str, Any]], attachments: AttachmentsConfig
) -> dict[int, int]:
    """What attachment CONTENT will cost when this replay set is hydrated.

    Returned per row index rather than as one total, because the cost has to be
    attributable: retention decides how much of the tail to keep by walking
    messages backwards and adding up what each one costs, and a photo that
    prices as its filename there would let the tail keep far more than its
    budget allows.

    Persisted rows carry only a reference to each file — `{id, name, mime,
    size, kind}`. The bytes and the extracted text are re-attached later, by
    `attachments.hydrate_history`, and only then does the request grow by an
    image block or a page of document text. Pricing the stored rows alone
    therefore misses the single most expensive thing in them: one photo costs
    around 800 tokens and reads as 15.

    That error runs in the unsafe direction — it makes a full conversation look
    roomy, so compaction waits. This function closes it by modelling what
    hydration will do, from the same configuration hydration itself obeys.

    Two different kinds of estimate, deliberately:

    - **Images are priced exactly.** `image_history_turns` decides how many of
      the most recent image-bearing turns are re-sent as pictures; the rest
      degrade to a one-line placeholder that `estimate_message` already counted.
      Walking newest-first mirrors `hydrate_history` exactly.
    - **Documents are priced by their allowance**, because knowing the real
      length would mean reading the attachment store — an I/O round trip per
      message, on a path that runs before every compaction check. `Budget`
      caps each document at `max_chars_per_doc` and the whole request at
      `max_chars_total`, spent newest-first, so the allowance is a true upper
      bound. A deployment whose `max_chars_total` is a large share of its
      `context_window` will therefore look fuller than it is; the fix there is
      to lower the injection budget, which lowers the real cost too.
    """
    image_turns_left = attachments.image_history_turns if attachments.vision_model else 0
    doc_chars_left = attachments.max_chars_total
    density = _observed_density(rows)
    per_row: dict[int, int] = {}

    for index in range(len(rows) - 1, -1, -1):
        row = rows[index]
        content = row.get("content")
        if not isinstance(content, dict):
            continue
        # An already-hydrated row carries its real content in `blocks`, which
        # `estimate_message` priced. Pricing it again here would double-count.
        if content.get("blocks"):
            continue
        refs = [a for a in (content.get("attachments") or []) if isinstance(a, dict)]
        if not refs:
            continue

        cost = 0
        images = [a for a in refs if a.get("kind") == "image"]
        if images and image_turns_left > 0:
            cost += len(images) * _IMAGE_TOKENS
            image_turns_left -= 1

        for _doc in (a for a in refs if a.get("kind") != "image"):
            if doc_chars_left <= 0:
                break
            allowance = min(attachments.max_chars_per_doc, doc_chars_left)
            doc_chars_left -= allowance
            cost += int(allowance * density)

        if cost:
            per_row[index] = cost

    return per_row


def _observed_density(rows: list[dict[str, Any]]) -> float:
    """Tokens per character, measured on this conversation's own visible text.

    A character budget says nothing about token cost on its own: 40 000
    characters is roughly 10 000 tokens of English and roughly 36 000 of
    Chinese. Picking either constant is wrong for half the deployments, and
    tokenizing a synthetic filler string is worse than both — a run of one
    repeated character compresses to almost nothing and would model a document
    at a third of its real cost.

    Sampling the conversation sidesteps the guess: whatever language the people
    in it are writing, their documents are overwhelmingly in that language too.
    """
    chars = 0
    tokens = 0
    for row in rows:
        content = row.get("content")
        if not isinstance(content, dict):
            continue
        text = content.get("text")
        if isinstance(text, str) and text:
            chars += len(text)
            tokens += estimate_text(text)
        if chars >= 2_000:  # a large enough sample; stop paying to tokenize
            break
    if chars == 0:
        return _DEFAULT_DENSITY
    # Clamp: a tiny unrepresentative sample (one emoji, one URL) must not scale
    # a 40 000-character budget into nonsense in either direction.
    return min(1.0, max(0.2, tokens / chars))


def estimate_row_costs(
    rows: list[dict[str, Any]], *, attachments: AttachmentsConfig | None = None
) -> list[int]:
    """Per-row tokens for a replay set, positionally aligned with `rows`.

    Attachment content is charged to the row that carries the file, so a caller
    walking the list backwards to size a retained tail charges the same price
    the whole-set total does. Whether a file's content replays at all is a
    whole-set decision, which is why this cannot be computed one row at a time.

    Pass `attachments` when the deployment has uploads switched on. Omitting it
    prices the stored rows verbatim, which is correct where nothing is ever
    hydrated — and correct for the summarization input too, since that replays
    the stored rows rather than a hydrated request.
    """
    costs = [estimate_message(row) for row in rows]
    if attachments is not None:
        for index, extra in _replay_attachment_tokens(rows, attachments).items():
            costs[index] += extra
    return costs


def estimate_messages(
    rows: list[dict[str, Any]], *, attachments: AttachmentsConfig | None = None
) -> int:
    """Tokens for a whole replay set."""
    return sum(estimate_row_costs(rows, attachments=attachments))


def reset_encoder_cache() -> None:
    """Forget the resolved encoder. For tests that exercise both paths."""
    global _encoder, _encoder_tried
    with _encoder_lock:
        _encoder = None
        _encoder_tried = False
