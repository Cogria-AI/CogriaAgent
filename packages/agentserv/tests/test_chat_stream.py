"""/chat streaming semantics: the graph is decoupled from the SSE response.

Drives the ASGI app directly (scope/receive/send) so a client disconnect can be
injected mid-stream deterministically — the "user navigated away" path. The
three guarantees under test:

  1. disconnect mid-generation does NOT kill the turn: it runs to completion
     and the full assistant text persists (not a truncated one);
  2. the `done` frame is emitted only AFTER the turn is persisted, so a history
     fetch triggered by `done` can't race the write;
  3. on a continuation, the user's message persists at turn START, visible to a
     history fetch made while the reply is still streaming.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import cogria_agent.server as server_mod
import jwt
import pytest
from cogria_agent import (
    AgentConfig,
    AuthConfig,
    InMemoryConversationBackend,
    StaticCatalogProvider,
    build_app,
)
from cogria_agent.config import SummarizerConfig
from langchain_core.messages import AIMessage, AIMessageChunk

SECRET = "test-secret-please-ignore"


class _NoopExecutor:
    async def invoke(self, *, name, args, query=None, context=None):
        return {"ok": True, "data": {}}


class _FakeGraph:
    """Streams the given deltas as chat-model chunks, then ends the model turn."""

    def __init__(self, deltas: list[str]):
        self._deltas = deltas

    async def astream_events(self, _state, version):
        for d in self._deltas:
            await asyncio.sleep(0.01)
            yield {"event": "on_chat_model_stream", "data": {"chunk": AIMessageChunk(content=d)}}
        full = "".join(self._deltas)
        yield {"event": "on_chat_model_end", "data": {"output": AIMessage(content=full)}}


def _build(backend: InMemoryConversationBackend, monkeypatch, deltas: list[str]):
    monkeypatch.setattr(server_mod, "build_graph", lambda *a, **k: _FakeGraph(deltas))
    config = AgentConfig(
        auth=AuthConfig(jwt_secret=SECRET, jwt_issuer="cogria-agent"),
        summarizer=SummarizerConfig(enabled=False),
    )
    return build_app(
        config,
        conversation_backend=backend,
        action_executor=_NoopExecutor(),
        catalog_provider=StaticCatalogProvider({"actions": []}),
    )


def _token() -> str:
    payload = {"iss": "cogria-agent", "sub": "u1", "exp": int(time.time()) + 300}
    return jwt.encode(payload, SECRET, algorithm="HS256")


async def _post_chat(
    app,
    body: dict[str, Any],
    *,
    disconnect_after_text_frames: int | None = None,
    on_frame=None,
) -> list[str]:
    """Run POST /chat through the raw ASGI interface, returning the SSE frames
    received. `disconnect_after_text_frames=N` injects http.disconnect after the
    N-th `text` frame — starlette then cancels the response generator, exactly
    what a page navigation does. `on_frame(frame)` is awaited per frame."""
    payload = json.dumps(body).encode()
    frames: list[str] = []
    text_frames = 0
    disconnected = asyncio.Event()
    body_sent = False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal text_frames
        if message["type"] != "http.response.body":
            return
        chunk = (message.get("body") or b"").decode()
        if not chunk:
            return
        frames.append(chunk)
        if on_frame is not None:
            await on_frame(chunk)
        if chunk.startswith("event: text"):
            text_frames += 1
            if disconnect_after_text_frames is not None:
                if text_frames >= disconnect_after_text_frames:
                    disconnected.set()

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/chat",
        "raw_path": b"/chat",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"authorization", f"Bearer {_token()}".encode()),
            (b"content-type", b"application/json"),
        ],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    await app(scope, receive, send)
    return frames


def _only_conversation(backend: InMemoryConversationBackend) -> str:
    """The id of the single conversation this app created. Ids are opaque, so a
    test can no longer assume the first one is `1`."""
    ids = list(backend._store)
    assert len(ids) == 1, f"expected exactly one conversation, got {len(ids)}"
    return ids[0]


async def _wait_for_turn(backend: InMemoryConversationBackend, cid: str, count: int, timeout=2.0):
    """Poll until the conversation holds `count` messages (the background graph
    task persists after the client is gone)."""
    for _ in range(int(timeout / 0.01)):
        rows = await backend.fetch_history(cid, for_llm=False)
        if len(rows) >= count:
            return rows
        await asyncio.sleep(0.01)
    pytest.fail(f"conversation {cid} never reached {count} messages")


async def test_disconnect_midstream_turn_still_persists_in_full(monkeypatch):
    backend = InMemoryConversationBackend()
    app = _build(backend, monkeypatch, ["Hel", "lo ", "world"])

    frames = await _post_chat(app, {"message": "hi"}, disconnect_after_text_frames=1)

    # The client bailed early: it saw at most a fragment, certainly no `done`.
    assert not any(f.startswith("event: done") for f in frames)

    rows = await _wait_for_turn(backend, _only_conversation(backend), 2)
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[1]["content"]["text"] == "Hello world"
    assert rows[1].get("error") is None


async def test_done_frame_arrives_only_after_persist(monkeypatch):
    backend = InMemoryConversationBackend()
    app = _build(backend, monkeypatch, ["a", "b"])

    persisted_at_done: list[dict[str, Any]] | None = None

    async def on_frame(chunk: str):
        nonlocal persisted_at_done
        if chunk.startswith("event: done"):
            persisted_at_done = await backend.fetch_history(
                _only_conversation(backend), for_llm=False
            )

    await _post_chat(app, {"message": "hi"}, on_frame=on_frame)

    assert persisted_at_done is not None, "no done frame seen"
    assert [r["role"] for r in persisted_at_done] == ["user", "assistant"]
    assert persisted_at_done[1]["content"]["text"] == "ab"


async def test_continuation_user_message_visible_while_streaming(monkeypatch):
    backend = InMemoryConversationBackend()
    cid = await backend.create_conversation(first_message="hi", model=None, user_id="u1")
    app = _build(backend, monkeypatch, ["re", "ply"])

    seen_mid_stream: list[dict[str, Any]] | None = None

    async def on_frame(chunk: str):
        nonlocal seen_mid_stream
        if seen_mid_stream is None and chunk.startswith("event: text"):
            seen_mid_stream = await backend.fetch_history(cid, for_llm=False)

    await _post_chat(app, {"conversation_id": cid, "message": "again"}, on_frame=on_frame)

    assert seen_mid_stream is not None, "no text frame seen"
    # Mid-stream history already shows the user's new turn, not yet the reply.
    assert [r["role"] for r in seen_mid_stream] == ["user", "user"]
    assert seen_mid_stream[1]["content"]["text"] == "again"

    rows = await _wait_for_turn(backend, cid, 3)
    assert [r["role"] for r in rows] == ["user", "user", "assistant"]
    assert rows[2]["content"]["text"] == "reply"
