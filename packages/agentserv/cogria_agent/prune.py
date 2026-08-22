"""Model-free tool-result pruning, applied on the way into history.

A tool result is written once and then resent on every subsequent turn. An
action returning a large list — the common case for this framework, whose whole
premise is business actions returning rows — is therefore charged again and
again until compaction eventually folds it away. Trimming it before it lands
costs nothing: no model call, no judgement, just a bounded head and tail.

Three rules the implementation exists to enforce:

- **Propose/confirm envelopes are never touched.** They carry a
  `proposal_token` the model has to hand back verbatim; a truncation that split
  one would surface as "I confirmed and nothing happened". They are small
  anyway, so exempting them costs nothing.
- **The result stays valid JSON.** Envelopes are parsed objects by the time
  they reach persistence, so pruning shrinks a value inside `data` and leaves
  the envelope's own keys alone. `graph._is_proposal` and the model both keep
  reading a well-formed object.
- **Pruning is idempotent.** A pruned result is strictly smaller than its
  threshold, so a second pass finds nothing to do and returns None.

Only the persisted copy is pruned. The in-flight graph state keeps the full
result for the rest of the turn — the model usually needs it in the round that
requested it — which bounds the untrimmed cost at `max_turns` resends within
one turn, against unbounded resends across every later turn.
"""

from __future__ import annotations

import json
from typing import Any

#: Inserted where content was removed. Worded so the model understands the gap
#: is a harness decision, not a tool failure or an empty result.
OMISSION_MARKER = (
    "\n\n[… {omitted} characters of this tool result were omitted to save context …]\n\n"
)

#: Added beside a truncated list so the model can see the list was cut rather
#: than assume it received every row.
TRUNCATION_KEY = "_truncated"


def _is_exempt(payload: Any) -> bool:
    """True for a propose/confirm envelope, which must survive verbatim."""
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    if not isinstance(data, dict):
        return False
    return bool(data.get("requires_confirm") or data.get("proposal_token"))


def _size(payload: Any) -> int:
    """Serialized size of a result, in characters — what history actually pays."""
    if isinstance(payload, str):
        return len(payload)
    try:
        return len(json.dumps(payload, ensure_ascii=False))
    except (TypeError, ValueError):
        return len(str(payload))


def _cut_text(text: str, *, head: int, tail: int) -> str:
    """Head + marker + tail, with the omitted count named in the marker."""
    omitted = len(text) - head - tail
    if omitted <= 0:
        return text
    marker = OMISSION_MARKER.format(omitted=omitted)
    return text[:head] + marker + (text[len(text) - tail :] if tail else "")


def _prune_container(value: Any, *, keep: int) -> tuple[Any, dict[str, Any] | None] | None:
    """Shrink the longest list reachable in `value`.

    Returns `(shrunk, marker)` or None when there is no list to shrink. A
    `marker` means the caller must place it beside the shrunk value, because the
    value's own type has no room for it.

    The shrunk value keeps the type it had. Turning a list-valued `data` into
    `{"items": [...]}` to make space for the marker would silently change the
    shape the action documented and the model was taught to read.
    """
    if isinstance(value, list):
        if len(value) <= keep:
            return None
        return value[:keep], {"kept": keep, "omitted": len(value) - keep}
    if isinstance(value, dict):
        # Shrink the single biggest list; that is nearly always the row set.
        best_key, best_len = None, keep
        for key, item in value.items():
            if isinstance(item, list) and len(item) > best_len:
                best_key, best_len = key, len(item)
        if best_key is None:
            return None
        shrunk = dict(value)
        shrunk[best_key] = value[best_key][:keep]
        shrunk[TRUNCATION_KEY] = {
            "field": best_key,
            "kept": keep,
            "omitted": best_len - keep,
        }
        return shrunk, None
    return None


def prune_result(
    payload: Any,
    *,
    threshold_chars: int = 8_192,
    head_chars: int = 4_096,
    tail_chars: int = 1_024,
    keep_items: int = 20,
) -> Any | None:
    """Return a bounded replacement for an oversized tool result, or None.

    None means "leave it alone": already small enough, or a propose/confirm
    envelope that must not be altered. A returned replacement is always at most
    `threshold_chars` serialized characters and strictly smaller than the input,
    so re-pruning it is a no-op.
    """
    if _is_exempt(payload):
        return None
    if _size(payload) <= threshold_chars:
        return None

    # Plain text (a tool that returned a bare string, or a non-JSON result).
    if isinstance(payload, str):
        return _cut_text(payload, head=head_chars, tail=tail_chars)

    # Structured envelope: shrink inside `data`, leaving ok/error/message intact
    # so the shape the model was taught to read survives.
    if isinstance(payload, dict):
        for key in ("data", "result", "items"):
            if key not in payload:
                continue
            # The rows themselves may be oversized, so keeping the configured
            # number of them can still blow the budget; try progressively fewer.
            for keep in (keep_items, keep_items // 2, keep_items // 4, 1):
                if keep < 1:
                    continue
                shrunk = _prune_container(payload[key], keep=keep)
                if shrunk is None:
                    break  # nothing list-shaped here; no smaller `keep` will help
                value, marker = shrunk
                candidate = dict(payload)
                candidate[key] = value
                if marker is not None:
                    candidate[TRUNCATION_KEY] = marker
                if _size(candidate) <= threshold_chars:
                    return candidate
            break

    # Nothing structural worked: fall back to cutting the serialized form. The
    # replacement is a string rather than an object, which the model still reads
    # fine and which `history_to_messages` passes through unchanged.
    text = json.dumps(payload, ensure_ascii=False)
    return _cut_text(text, head=head_chars, tail=tail_chars)
