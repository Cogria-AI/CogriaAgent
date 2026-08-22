"""`history_search`: reading back what compaction folded away."""

from __future__ import annotations

import json

import pytest
from cogria_agent.config import RecallConfig
from cogria_agent.inmemory import InMemoryConversationBackend
from cogria_agent.recall import build_recall_tool, search_rows


def _rows() -> list[dict]:
    return [
        {"role": "user", "content": {"text": "my order is ORD-99123"}},
        {
            "role": "assistant",
            "content": {
                "text": "",
                "tool_calls": [
                    {"id": "c1", "name": "lookup", "args": {"id": "ORD-99123"}}
                ],
            },
        },
        {
            "role": "tool",
            "content": {"tool_call_id": "c1", "name": "lookup", "result": {"total": "42.50"}},
        },
        {"role": "assistant", "content": {"text": "It totals 42.50."}},
    ]


def test_finds_a_term_in_message_text():
    hits = search_rows(_rows(), "ORD-99123", limit=5, snippet_chars=200)
    assert hits and any(h["role"] == "user" for h in hits)


def test_search_covers_tool_arguments_and_results():
    """The exact value the model needs back is as likely to be inside a tool
    result as in prose — that is precisely what a checkpoint summarises away."""
    assert search_rows(_rows(), "42.50", limit=5, snippet_chars=200)
    assert search_rows(_rows(), "lookup", limit=5, snippet_chars=200)


def test_matching_is_case_insensitive():
    assert search_rows(_rows(), "ord-99123", limit=5, snippet_chars=200)


def test_newest_first():
    rows = [{"role": "user", "content": {"text": f"ping {i}"}} for i in range(10)]
    hits = search_rows(rows, "ping", limit=3, snippet_chars=50)
    assert [h["position"] for h in hits] == [9, 8, 7]


def test_limit_is_respected():
    rows = [{"role": "user", "content": {"text": "ping"}} for _ in range(50)]
    assert len(search_rows(rows, "ping", limit=4, snippet_chars=50)) == 4


def test_empty_query_matches_nothing():
    assert search_rows(_rows(), "   ", limit=5, snippet_chars=200) == []


def test_snippet_is_bounded_and_marks_elision():
    rows = [{"role": "user", "content": {"text": "a" * 500 + "NEEDLE" + "b" * 500}}]
    hit = search_rows(rows, "NEEDLE", limit=1, snippet_chars=100)[0]
    assert "NEEDLE" in hit["excerpt"]
    assert len(hit["excerpt"]) <= 102
    assert hit["excerpt"].startswith("…")


@pytest.mark.asyncio
async def test_tool_reads_folded_messages():
    """The point of the tool: `summarized_count` is a cursor, not a delete, so
    a value that is no longer replayed is still reachable."""
    backend = InMemoryConversationBackend()
    cid = await backend.create_conversation(first_message="invoice INV-777 please", model="m")
    await backend.append_messages(
        cid,
        messages=[{"role": "assistant", "content": {"text": f"ok {i}"}} for i in range(6)],
        usage=None,
        model="m",
    )
    await backend.save_summary(cid, summary="CHECKPOINT", through_index=5)

    # The folded message is gone from the replay set...
    replay = await backend.fetch_history(cid, for_llm=True)
    assert not any("INV-777" in json.dumps(r, ensure_ascii=False) for r in replay)

    # ...and still reachable through the tool.
    tool = build_recall_tool(backend=backend, conversation_id=cid, config=RecallConfig())
    envelope = json.loads(await tool.ainvoke({"query": "INV-777"}))
    assert envelope["ok"] is True
    assert envelope["data"]["matches"]
    assert "INV-777" in envelope["data"]["matches"][0]["excerpt"]


@pytest.mark.asyncio
async def test_tool_is_bound_to_one_conversation():
    """The id is closed over, never taken from the model, so reading another
    conversation is unreachable rather than merely unauthorized."""
    backend = InMemoryConversationBackend()
    mine = await backend.create_conversation(first_message="mine SECRET-A", model="m")
    await backend.create_conversation(first_message="theirs SECRET-B", model="m")

    tool = build_recall_tool(backend=backend, conversation_id=mine, config=RecallConfig())
    assert "query" in tool.args_schema.model_fields
    assert set(tool.args_schema.model_fields) == {"query"}

    envelope = json.loads(await tool.ainvoke({"query": "SECRET-B"}))
    assert envelope["ok"] is True
    assert envelope["data"]["matches"] == []


@pytest.mark.asyncio
async def test_backend_failure_does_not_break_chat():
    class _Broken:
        async def fetch_messages_full(self, conversation_id):
            raise RuntimeError("database gone")

    tool = build_recall_tool(backend=_Broken(), conversation_id="c", config=RecallConfig())
    envelope = json.loads(await tool.ainvoke({"query": "anything"}))
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "HISTORY_SEARCH_FAILED"


def test_recall_is_off_by_default():
    """An optional capability costs nothing until asked for: a mounted tool adds
    its schema to every request."""
    assert RecallConfig().enabled is False


@pytest.mark.asyncio
async def test_tool_is_mounted_only_when_enabled(monkeypatch):
    """Off by default means the schema is absent from the request, not merely
    unused — a mounted tool is paid for on every turn."""
    import time

    import cogria_agent.server as server_mod
    import jwt
    from cogria_agent import AgentConfig, AuthConfig, StaticCatalogProvider, build_app
    from cogria_agent.config import SummarizerConfig
    from langchain_core.messages import AIMessage, AIMessageChunk

    secret = "test-secret-please-ignore"

    class _Graph:
        async def astream_events(self, _state, version):
            yield {"event": "on_chat_model_stream", "data": {"chunk": AIMessageChunk(content="k")}}
            yield {"event": "on_chat_model_end", "data": {"output": AIMessage(content="k")}}

    class _Noop:
        async def invoke(self, *, name, args, query=None, context=None):
            return {"ok": True, "data": {}}

    monkeypatch.setattr(server_mod, "build_graph", lambda *a, **k: _Graph())

    async def _ready_frame_tools(recall_enabled: bool) -> list[str]:
        app = build_app(
            AgentConfig(
                auth=AuthConfig(jwt_secret=secret, jwt_issuer="cogria-agent"),
                summarizer=SummarizerConfig(enabled=False),
                recall=RecallConfig(enabled=recall_enabled),
            ),
            conversation_backend=InMemoryConversationBackend(),
            action_executor=_Noop(),
            catalog_provider=StaticCatalogProvider({"actions": []}),
        )
        frames: list[str] = []
        sent = False
        token = jwt.encode(
            {"iss": "cogria-agent", "sub": "u1", "exp": int(time.time()) + 300},
            secret,
            algorithm="HS256",
        )

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": b'{"message":"hi"}', "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                frames.append(message["body"].decode())

        await app(
            {
                "type": "http", "http_version": "1.1", "method": "POST", "scheme": "http",
                "path": "/chat", "raw_path": b"/chat", "query_string": b"", "root_path": "",
                "headers": [
                    (b"authorization", f"Bearer {token}".encode()),
                    (b"content-type", b"application/json"),
                ],
                "client": ("testclient", 50000), "server": ("testserver", 80),
            },
            receive,
            send,
        )
        ready = next(f for f in frames if f.startswith("event: ready"))
        return json.loads(ready.split("data: ", 1)[1])["tools"]

    assert "history_search" not in await _ready_frame_tools(False)
    assert "history_search" in await _ready_frame_tools(True)
