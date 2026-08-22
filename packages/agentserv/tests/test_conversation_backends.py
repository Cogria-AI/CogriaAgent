"""One behaviour suite, both ConversationBackend implementations.

The kernel is written against the protocol, not against a backend, so the
in-memory and SQL backends have to be indistinguishable from where the kernel
sits. Everything here runs twice; the SQL cases additionally prove rows survive
a fresh backend instance (the whole point of C7.0).

SQL cases skip when the `sql` extra isn't installed.
"""

from __future__ import annotations

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy", reason="needs the `sql` extra")
pytest.importorskip("aiosqlite", reason="needs the `sql` extra")

from cogria_agent.inmemory import InMemoryConversationBackend  # noqa: E402
from cogria_agent.sqlbackend import SqlConversationBackend  # noqa: E402


@pytest.fixture(params=["inmemory", "sql"])
async def backend(request, tmp_path):
    if request.param == "inmemory":
        yield InMemoryConversationBackend()
        return
    b = SqlConversationBackend(f"sqlite+aiosqlite:///{tmp_path / 'agent.db'}")
    await b.create_all()
    try:
        yield b
    finally:
        await b.dispose()


async def test_create_persists_first_user_message(backend):
    cid = await backend.create_conversation(first_message="hello", model="m1")
    rows = await backend.fetch_history(cid)
    assert len(rows) == 1
    assert rows[0]["role"] == "user"
    assert rows[0]["content"]["text"] == "hello"


async def test_first_content_carries_richer_shape(backend):
    content = {"text": "look", "attachments": [{"id": "att_x", "name": "a.pdf"}]}
    cid = await backend.create_conversation(first_message="look", model="m1", first_content=content)
    rows = await backend.fetch_history(cid)
    # Unknown attachment ids must not break the write — the JSON copy is kept
    # verbatim either way (the link table only references rows that exist).
    assert rows[0]["content"]["attachments"][0]["id"] == "att_x"


async def test_append_keeps_order_and_totals(backend):
    cid = await backend.create_conversation(first_message="q", model="m1")
    await backend.append_messages(
        cid,
        messages=[
            {"role": "assistant", "content": {"text": "a1"}, "finish_reason": "stop"},
            {
                "role": "tool",
                "content": {"tool_call_id": "c1", "name": "t", "result": {"ok": True}},
            },
        ],
        usage={"input_tokens": 10, "output_tokens": 4},
        model="m1",
    )
    await backend.append_messages(
        cid,
        messages=[{"role": "user", "content": {"text": "q2"}}],
        usage={"input_tokens": 5, "output_tokens": 1},
        model="m1",
    )

    rows = await backend.fetch_history(cid)
    assert [r["role"] for r in rows] == ["user", "assistant", "tool", "user"]
    assert rows[2]["content"]["result"] == {"ok": True}

    meta = await backend.fetch_meta(cid)
    assert meta["message_count"] == 4
    assert meta["total_input_tokens"] == 15
    assert meta["total_output_tokens"] == 5
    assert meta["model"] == "m1"


async def test_append_to_unknown_conversation_raises(backend):
    with pytest.raises(KeyError):
        await backend.append_messages(
            99999, messages=[{"role": "user", "content": {}}], usage=None, model=None
        )


async def test_unknown_conversation_reads_empty(backend):
    assert await backend.fetch_history(99999) == []
    assert await backend.fetch_meta(99999) == {}


async def test_summary_replaces_folded_head(backend):
    cid = await backend.create_conversation(first_message="m0", model="m1")
    await backend.append_messages(
        cid,
        messages=[{"role": "assistant", "content": {"text": f"m{i}"}} for i in range(1, 8)],
        usage=None,
        model="m1",
    )
    await backend.save_summary(cid, summary="SUMMARY", through_index=5)

    replay = await backend.fetch_history(cid, for_llm=True)
    # The checkpoint replays as a USER message: the operating prompt is the only
    # system-role instruction the model should receive, and a summary sitting in
    # that role reads as one.
    assert replay[0]["role"] == "user" and replay[0]["content"]["text"] == "SUMMARY"
    assert [r["content"]["text"] for r in replay[1:]] == ["m5", "m6", "m7"]
    # fetch_meta exposes the running checkpoint so the next compaction can merge
    # into it rather than produce a second, overlapping one.
    assert (await backend.fetch_meta(cid))["summary"] == "SUMMARY"

    # for_llm=False is the raw record — no summary head, nothing dropped.
    full = await backend.fetch_history(cid, for_llm=False)
    assert len(full) == 8 and full[0]["content"]["text"] == "m0"
    assert len(await backend.fetch_messages_full(cid)) == 8
    assert (await backend.fetch_meta(cid))["summarized_count"] == 5


async def test_sql_rows_survive_a_new_backend_instance(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'agent.db'}"
    first = SqlConversationBackend(url)
    await first.create_all()
    cid = await first.create_conversation(first_message="remember me", model="m1")
    await first.append_messages(
        cid, messages=[{"role": "assistant", "content": {"text": "sure"}}], usage=None, model="m1"
    )
    await first.dispose()

    # New process, same file.
    second = SqlConversationBackend(url)
    try:
        rows = await second.fetch_history(cid)
        assert [r["content"]["text"] for r in rows] == ["remember me", "sure"]
        assert (await second.fetch_meta(cid))["title"] == "remember me"
    finally:
        await second.dispose()


async def test_sql_seq_stays_dense_under_concurrent_appends(tmp_path):
    """Two turns landing at once must not collide on (conversation_id, seq)."""
    import asyncio

    b = SqlConversationBackend(f"sqlite+aiosqlite:///{tmp_path / 'agent.db'}")
    await b.create_all()
    try:
        cid = await b.create_conversation(first_message="start", model="m1")
        await asyncio.gather(
            *(
                b.append_messages(
                    cid,
                    messages=[{"role": "assistant", "content": {"text": f"r{i}"}}],
                    usage=None,
                    model="m1",
                )
                for i in range(8)
            )
        )
        rows = await b.fetch_history(cid)
        assert len(rows) == 9
        assert sorted(r["content"]["text"] for r in rows[1:]) == [f"r{i}" for i in range(8)]
    finally:
        await b.dispose()
