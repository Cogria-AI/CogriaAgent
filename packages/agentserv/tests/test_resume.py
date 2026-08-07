"""Resumable runs: re-attaching to a reply that is still being generated.

The scenario is a user who sends a message, navigates away mid-reply, and comes
back. Three things have to hold, and each is easy to break:

  1. the turn survives the disconnect (covered in test_chat_stream.py),
  2. a returning client can replay what it missed AND follow the rest live,
  3. ownership is enforced on the replay stream exactly as on the read routes.

The disconnect is injected through the raw ASGI interface so it lands at a
deterministic point rather than whenever a timer happens to fire.
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


class _GatedGraph:
    """Streams deltas, pausing on a gate so a test can hold a run open."""

    def __init__(self, deltas: list[str], gate: asyncio.Event | None, gate_after: int):
        self._deltas = deltas
        self._gate = gate
        self._gate_after = gate_after

    async def astream_events(self, _state, version):
        for i, d in enumerate(self._deltas):
            if self._gate is not None and i == self._gate_after:
                await self._gate.wait()
            await asyncio.sleep(0)
            yield {"event": "on_chat_model_stream", "data": {"chunk": AIMessageChunk(content=d)}}
        yield {
            "event": "on_chat_model_end",
            "data": {"output": AIMessage(content="".join(self._deltas))},
        }


def _build(backend, monkeypatch, deltas, *, gate=None, gate_after=1):
    monkeypatch.setattr(
        server_mod, "build_graph", lambda *a, **k: _GatedGraph(deltas, gate, gate_after)
    )
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


def _token(sub="u1") -> str:
    return jwt.encode(
        {"iss": "cogria-agent", "sub": sub, "exp": int(time.time()) + 300},
        SECRET,
        algorithm="HS256",
    )


def _scope(method: str, path: str, *, sub="u1", query: bytes = b""):
    return {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query,
        "root_path": "",
        "headers": [
            (b"authorization", f"Bearer {_token(sub)}".encode()),
            (b"content-type", b"application/json"),
        ],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }


def _frames(chunks: list[str]) -> list[tuple[str, dict]]:
    """Parse accumulated SSE text into (event, payload) pairs."""
    out = []
    for block in "".join(chunks).split("\n\n"):
        if not block.startswith("event: "):
            continue
        name = block.split("\n", 1)[0][len("event: ") :]
        data = block.split("data: ", 1)[1] if "data: " in block else "{}"
        out.append((name, json.loads(data)))
    return out


async def _drive(app, scope, body: dict | None, *, stop_after_text: int | None = None):
    """Run one ASGI request, optionally disconnecting after N `text` frames.
    Returns (chunks, status)."""
    payload = json.dumps(body or {}).encode()
    chunks: list[str] = []
    status = {"code": 0}
    seen_text = 0
    disconnected = asyncio.Event()
    sent_body = False

    async def receive():
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal seen_text
        if message["type"] == "http.response.start":
            status["code"] = message["status"]
            return
        if message["type"] != "http.response.body":
            return
        text = (message.get("body") or b"").decode()
        if not text:
            return
        chunks.append(text)
        if text.startswith("event: text"):
            seen_text += 1
            if stop_after_text is not None and seen_text >= stop_after_text:
                disconnected.set()

    await app(scope, receive, send)
    return chunks, status["code"]


async def test_returning_client_replays_the_backlog_and_follows_live(monkeypatch):
    """The core promise: leave mid-reply, come back, see the whole reply."""
    backend = InMemoryConversationBackend()
    gate = asyncio.Event()
    app = _build(backend, monkeypatch, ["Hel", "lo ", "world"], gate=gate, gate_after=1)

    # Send a message, then walk away after the first delta.
    first, _ = await _drive(app, _scope("POST", "/chat"), {"message": "hi"}, stop_after_text=1)
    assert [e for e, _ in _frames(first)].count("text") == 1
    cid = next(p["conversation_id"] for e, p in _frames(first) if e == "conversation")

    # The run is still open and the server knows it.
    detail, code = await _drive(app, _scope("GET", f"/conversations/{cid}"), None)
    assert code == 200
    assert json.loads("".join(detail))["active"] is True

    # Come back: replay + live tail. Release the gate once we're attached.
    async def release():
        await asyncio.sleep(0.05)
        gate.set()

    asyncio.ensure_future(release())
    resumed, code = await _drive(app, _scope("GET", f"/conversations/{cid}/stream"), None)
    assert code == 200

    events = _frames(resumed)
    deltas = "".join(p["delta"] for e, p in events if e == "text")
    # The delta emitted before the client returned (replay) plus the ones after
    # it attached (live) — the whole reply, exactly once.
    assert deltas == "Hello world"
    assert events[0][0] == "conversation" and events[0][1]["new"] is False
    assert events[-1][0] == "done"
    # A resume is a continuation, not a fresh turn: no tool advertisement.
    assert "ready" not in [e for e, _ in events]


async def test_no_frame_is_duplicated_between_backlog_and_live_tail(monkeypatch):
    """The subscribe/snapshot pair must be atomic; if it ever gains an await in
    between, a frame lands in both halves and the user sees doubled text."""
    backend = InMemoryConversationBackend()
    gate = asyncio.Event()
    deltas = [f"d{i}" for i in range(8)]
    app = _build(backend, monkeypatch, deltas, gate=gate, gate_after=4)

    first, _ = await _drive(app, _scope("POST", "/chat"), {"message": "hi"}, stop_after_text=1)
    cid = next(p["conversation_id"] for e, p in _frames(first) if e == "conversation")

    async def release():
        await asyncio.sleep(0.05)
        gate.set()

    asyncio.ensure_future(release())
    resumed, _ = await _drive(app, _scope("GET", f"/conversations/{cid}/stream"), None)

    got = [p["delta"] for e, p in _frames(resumed) if e == "text"]
    assert got == deltas, f"expected each delta exactly once, got {got}"


async def test_resuming_a_finished_run_is_404(monkeypatch):
    """Nothing in flight: the client refetches history instead of streaming."""
    backend = InMemoryConversationBackend()
    app = _build(backend, monkeypatch, ["done"], gate=None)

    first, _ = await _drive(app, _scope("POST", "/chat"), {"message": "hi"})
    cid = next(p["conversation_id"] for e, p in _frames(first) if e == "conversation")
    await asyncio.sleep(0.05)  # let the run retract itself

    _, code = await _drive(app, _scope("GET", f"/conversations/{cid}/stream"), None)
    assert code == 404

    detail, _ = await _drive(app, _scope("GET", f"/conversations/{cid}"), None)
    assert json.loads("".join(detail))["active"] is False


async def test_the_registry_is_emptied_when_a_run_finishes(monkeypatch):
    """A leaked entry would leave a conversation permanently 'generating'."""
    backend = InMemoryConversationBackend()
    app = _build(backend, monkeypatch, ["a"], gate=None)
    await _drive(app, _scope("POST", "/chat"), {"message": "hi"})
    await asyncio.sleep(0.05)

    detail, _ = await _drive(
        app, _scope("GET", "/conversations"), None
    )
    listed = json.loads("".join(detail))["conversations"]
    assert len(listed) == 1
    cid = listed[0]["id"]
    _, code = await _drive(app, _scope("GET", f"/conversations/{cid}/stream"), None)
    assert code == 404, "the run should have retracted itself from the registry"


async def test_a_stranger_cannot_attach_to_someone_elses_run(monkeypatch):
    """Ownership is enforced on the stream, and indistinguishably from a missing
    conversation so an id can't be probed for existence."""
    backend = InMemoryConversationBackend()
    gate = asyncio.Event()
    app = _build(backend, monkeypatch, ["a", "b"], gate=gate, gate_after=1)

    first, _ = await _drive(
        app, _scope("POST", "/chat", sub="u1"), {"message": "hi"}, stop_after_text=1
    )
    cid = next(p["conversation_id"] for e, p in _frames(first) if e == "conversation")

    _, code = await _drive(app, _scope("GET", f"/conversations/{cid}/stream", sub="u2"), None)
    assert code == 404
    _, code = await _drive(app, _scope("GET", f"/conversations/{cid}", sub="u2"), None)
    assert code == 404

    gate.set()
    await asyncio.sleep(0.05)


async def test_a_stranger_cannot_continue_someone_elses_conversation(monkeypatch):
    """Continuation is a read (history into the prompt) and a write (append)."""
    backend = InMemoryConversationBackend()
    app = _build(backend, monkeypatch, ["a"], gate=None)

    first, _ = await _drive(app, _scope("POST", "/chat", sub="u1"), {"message": "mine"})
    cid = next(p["conversation_id"] for e, p in _frames(first) if e == "conversation")
    await asyncio.sleep(0.05)

    _, code = await _drive(
        app, _scope("POST", "/chat", sub="u2"), {"conversation_id": cid, "message": "sneak"}
    )
    assert code == 404

    rows = await backend.fetch_history(cid, for_llm=False)
    assert all("sneak" not in str(r.get("content")) for r in rows)


async def test_a_soft_deleted_conversation_cannot_be_resumed_or_continued(monkeypatch):
    backend = InMemoryConversationBackend()
    app = _build(backend, monkeypatch, ["a"], gate=None)

    first, _ = await _drive(app, _scope("POST", "/chat"), {"message": "hi"})
    cid = next(p["conversation_id"] for e, p in _frames(first) if e == "conversation")
    await asyncio.sleep(0.05)

    _, code = await _drive(app, _scope("DELETE", f"/conversations/{cid}"), None)
    assert code == 200

    for method, path in [
        ("GET", f"/conversations/{cid}"),
        ("GET", f"/conversations/{cid}/stream"),
        ("PATCH", f"/conversations/{cid}"),
    ]:
        _, code = await _drive(app, _scope(method, path), None)
        assert code == 404, f"{method} {path} should be gone"

    _, code = await _drive(
        app, _scope("POST", "/chat"), {"conversation_id": cid, "message": "again"}
    )
    assert code == 404


async def test_rename_is_owner_only_and_shows_up_in_the_list(monkeypatch):
    backend = InMemoryConversationBackend()
    app = _build(backend, monkeypatch, ["a"], gate=None)

    first, _ = await _drive(app, _scope("POST", "/chat"), {"message": "opening words"})
    cid = next(p["conversation_id"] for e, p in _frames(first) if e == "conversation")
    await asyncio.sleep(0.05)

    # A stranger's rename is the same 404 as a missing conversation.
    _, code = await _drive(
        app, _scope("PATCH", f"/conversations/{cid}", sub="u2"), {"title": "hijack"}
    )
    assert code == 404

    # Valid JSON that isn't an object must be a 422, not a 500.
    _, code = await _drive(app, _scope("PATCH", f"/conversations/{cid}"), ["not", "a", "dict"])
    assert code == 422

    body, code = await _drive(
        app, _scope("PATCH", f"/conversations/{cid}"), {"title": "  My   chat  "}
    )
    assert code == 200
    assert json.loads("".join(body))["title"] == "My chat"  # whitespace collapsed

    listed, _ = await _drive(app, _scope("GET", "/conversations"), None)
    rows = json.loads("".join(listed))["conversations"]
    assert rows[0]["title"] == "My chat"

    # Blank clears the stored title; the derived one comes back.
    _, code = await _drive(app, _scope("PATCH", f"/conversations/{cid}"), {"title": "   "})
    assert code == 200
    listed, _ = await _drive(app, _scope("GET", "/conversations"), None)
    rows = json.loads("".join(listed))["conversations"]
    assert rows[0]["title"] == "opening words"


async def test_two_clients_watching_one_run_both_see_everything(monkeypatch):
    """The POST response and a resume stream are just two subscribers."""
    backend = InMemoryConversationBackend()
    gate = asyncio.Event()
    deltas = ["x", "y", "z"]
    app = _build(backend, monkeypatch, deltas, gate=gate, gate_after=1)

    started: dict[str, Any] = {}

    async def poster():
        chunks, _ = await _drive(app, _scope("POST", "/chat"), {"message": "hi"})
        started["poster"] = chunks

    task = asyncio.ensure_future(poster())
    # Wait until the run registers, then attach a second watcher.
    for _ in range(100):
        await asyncio.sleep(0.01)
        listed, _ = await _drive(app, _scope("GET", "/conversations"), None)
        rows = json.loads("".join(listed))["conversations"]
        if rows:
            break
    cid = rows[0]["id"]

    async def release():
        await asyncio.sleep(0.05)
        gate.set()

    asyncio.ensure_future(release())
    watcher, _ = await _drive(app, _scope("GET", f"/conversations/{cid}/stream"), None)
    await task

    poster_text = "".join(p["delta"] for e, p in _frames(started["poster"]) if e == "text")
    watcher_text = "".join(p["delta"] for e, p in _frames(watcher) if e == "text")
    assert poster_text == "".join(deltas)
    assert watcher_text == "".join(deltas)


async def test_history_shows_the_user_turn_while_the_reply_is_still_running(monkeypatch):
    """What the returning client renders before the resume stream attaches."""
    backend = InMemoryConversationBackend()
    gate = asyncio.Event()
    app = _build(backend, monkeypatch, ["a", "b"], gate=gate, gate_after=1)

    # Walk away mid-reply; the run stays gated open behind us.
    first, _ = await _drive(
        app, _scope("POST", "/chat"), {"message": "remember me"}, stop_after_text=1
    )
    cid = next(p["conversation_id"] for e, p in _frames(first) if e == "conversation")

    detail, code = await _drive(app, _scope("GET", f"/conversations/{cid}"), None)
    assert code == 200
    body = json.loads("".join(detail))
    assert body["messages"][0]["content"]["text"] == "remember me"
    assert body["conversation"]["title"] == "remember me"

    gate.set()
    await asyncio.sleep(0.05)


@pytest.mark.parametrize("bad_id", ["not-a-uuid", "00000000-0000-4000-8000-000000000000", "1"])
async def test_unknown_conversation_ids_are_404_not_500(monkeypatch, bad_id):
    backend = InMemoryConversationBackend()
    app = _build(backend, monkeypatch, ["a"], gate=None)
    for path in [f"/conversations/{bad_id}", f"/conversations/{bad_id}/stream"]:
        _, code = await _drive(app, _scope("GET", path), None)
        assert code == 404, f"{path} -> {code}"
