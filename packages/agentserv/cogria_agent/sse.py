"""SSE framing + proposal extraction helpers.

The custom SSE protocol is the wire format between the BFF and this kernel:
  ready / conversation / text / tool_call / tool_result / confirm_required / error / done
"""

from __future__ import annotations

import json
from typing import Any


def sse(event: str, payload: dict[str, Any]) -> bytes:
    """Format one SSE frame. Explicit `event:` helps debugging."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def extract_proposal(content: Any) -> dict[str, Any] | None:
    """Return {proposal_token, summary} if a tool result is a propose (dry_run)
    envelope, else None. Tolerant of non-JSON content."""
    try:
        env = json.loads(content) if isinstance(content, str) else content
    except (TypeError, ValueError):
        return None
    if not isinstance(env, dict) or not env.get("ok"):
        return None
    data = env.get("data") or {}
    if isinstance(data, dict) and data.get("requires_confirm") and data.get("proposal_token"):
        return {"proposal_token": data.get("proposal_token"), "summary": data.get("summary", "")}
    return None
