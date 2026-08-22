"""Recovery when the provider rejects a request for exceeding its context window.

Before this path existed the rejection surfaced to the user as a raw provider
error, and the fire-and-forget compaction that might have prevented it had
already failed silently. The contract now is: compact, retry once, and if that
still doesn't fit, report the original error honestly.
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
from cogria_agent.config import GraphConfig, SummarizerConfig
from cogria_agent.server import _is_context_overflow
from langchain_core.messages import AIMessage, AIMessageChunk

SECRET = "test-secret-please-ignore"


class _NoopExecutor:
    async def invoke(self, *, name, args, query=None, context=None):
        return {"ok": True, "data": {}}


class _OverflowThenSucceed:
    """Rejects the first run as too long, then answers normally.

    Mirrors the real shape of the failure: the provider refuses before any token
    is streamed, so a retry cannot duplicate visible output.
    """

    def __init__(self, failures: int = 1, message: str | None = None) -> None:
        self.remaining = failures
        self.runs = 0
        self._message = message or (
            "This model's maximum context length is 8192 tokens. "
            "However, your messages resulted in 9000 tokens."
        )

    async def astream_events(self, _state, version):
        self.runs += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise RuntimeError(self._message)
        yield {"event": "on_chat_model_stream", "data": {"chunk": AIMessageChunk(content="hi")}}
        yield {"event": "on_chat_model_end", "data": {"output": AIMessage(content="hi")}}


class _FakeSummaryLLM:
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content="CHECKPOINT TEXT")


class _FakeLLMFactory:
    def chat_llm(self, *, model=None):
        raise AssertionError("the graph is stubbed; chat_llm must not be built")

    def summary_llm(self):
        return _FakeSummaryLLM()

    def model_name(self):
        return "fake-model"


def _token() -> str:
    return jwt.encode(
        {"iss": "cogria-agent", "sub": "u1", "exp": int(time.time()) + 300},
        SECRET,
        algorithm="HS256",
    )


async def _post_chat(app, body: dict[str, Any]) -> list[str]:
    payload = json.dumps(body).encode()
    frames: list[str] = []
    sent = False
    idle = asyncio.Event()

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await idle.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] != "http.response.body":
            return
        chunk = (message.get("body") or b"").decode()
        if chunk:
            frames.append(chunk)

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


def _build(backend, monkeypatch, graph, *, retries: int = 1):
    monkeypatch.setattr(server_mod, "build_graph", lambda *a, **k: graph)
    config = AgentConfig(
        auth=AuthConfig(jwt_secret=SECRET, jwt_issuer="cogria-agent"),
        graph=GraphConfig(max_overflow_retries=retries),
        # A tiny window so ordinary pressure compaction stays out of the way and
        # only the recovery path is under test.
        summarizer=SummarizerConfig(context_window=1_000_000),
    )
    return build_app(
        config,
        conversation_backend=backend,
        action_executor=_NoopExecutor(),
        catalog_provider=StaticCatalogProvider({"actions": []}),
        llm_factory=_FakeLLMFactory(),
    )


async def _seed(backend) -> str:
    cid = await backend.create_conversation(first_message="hello", model="m", user_id="u1")
    await backend.append_messages(
        cid,
        messages=[{"role": "assistant", "content": {"text": f"reply {i} " + "word " * 200}}
                  for i in range(10)],
        usage=None,
        model="m",
    )
    return cid


@pytest.mark.parametrize(
    "message",
    [
        "context_length_exceeded",
        "This model's maximum context length is 8192 tokens",
        "Input length and `max_tokens` exceed context limit",
        "prompt is too long: 300000 tokens > 200000 maximum",
    ],
)
def test_recognises_how_each_gateway_words_it(message):
    """No shared status code exists, so text is the only portable signal."""
    assert _is_context_overflow(RuntimeError(message))


def test_recognises_a_wrapped_cause():
    """The SDK error that names the reason is usually wrapped by the time
    LangChain re-raises it."""
    inner = RuntimeError("context_length_exceeded")
    outer = ValueError("request failed")
    outer.__cause__ = inner
    assert _is_context_overflow(outer)


def test_does_not_fire_on_unrelated_errors():
    """A false positive costs a pointless compaction plus a retry."""
    assert not _is_context_overflow(RuntimeError("rate limit exceeded"))
    assert not _is_context_overflow(RuntimeError("invalid api key"))
    assert not _is_context_overflow(RuntimeError("max_tokens must be an integer"))


@pytest.mark.asyncio
async def test_overflow_compacts_and_retries(monkeypatch):
    backend = InMemoryConversationBackend()
    cid = await _seed(backend)
    graph = _OverflowThenSucceed(failures=1)
    app = _build(backend, monkeypatch, graph)

    frames = await _post_chat(app, {"message": "and now?", "conversation_id": cid})
    joined = "".join(frames)

    assert graph.runs == 2  # rejected once, retried once
    assert "event: compacting" in joined
    assert "event: error" not in joined
    assert "hi" in joined

    # The retry ran against a genuinely compacted history.
    meta = await backend.fetch_meta(cid)
    assert meta["summarized_count"] > 0
    assert "CHECKPOINT TEXT" in (meta["summary"] or "")


@pytest.mark.asyncio
async def test_retry_budget_is_respected(monkeypatch):
    """If a maximally compacted history still doesn't fit, another attempt
    won't change that — report the failure instead of looping."""
    backend = InMemoryConversationBackend()
    cid = await _seed(backend)
    graph = _OverflowThenSucceed(failures=99)
    app = _build(backend, monkeypatch, graph, retries=1)

    frames = await _post_chat(app, {"message": "and now?", "conversation_id": cid})
    joined = "".join(frames)

    assert graph.runs == 2  # the original plus one retry, then it stops
    assert "event: error" in joined
    # The user gets the provider's real reason, not a harness-invented one.
    assert "maximum context length" in joined


@pytest.mark.asyncio
async def test_retries_are_disabled_by_configuration(monkeypatch):
    backend = InMemoryConversationBackend()
    cid = await _seed(backend)
    graph = _OverflowThenSucceed(failures=1)
    app = _build(backend, monkeypatch, graph, retries=0)

    frames = await _post_chat(app, {"message": "and now?", "conversation_id": cid})
    assert graph.runs == 1
    assert "event: error" in "".join(frames)


@pytest.mark.asyncio
async def test_unrelated_failures_are_not_retried(monkeypatch):
    class _AlwaysBoom:
        def __init__(self):
            self.runs = 0

        async def astream_events(self, _state, version):
            self.runs += 1
            raise RuntimeError("upstream is on fire")
            yield  # pragma: no cover — makes this an async generator

    backend = InMemoryConversationBackend()
    cid = await _seed(backend)
    graph = _AlwaysBoom()
    app = _build(backend, monkeypatch, graph)

    frames = await _post_chat(app, {"message": "and now?", "conversation_id": cid})
    assert graph.runs == 1
    assert "event: error" in "".join(frames)
