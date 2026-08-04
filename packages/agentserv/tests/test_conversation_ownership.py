"""Ownership, public ids and listing — run against BOTH ConversationBackends.

The two implementations are used interchangeably (in-memory for dev/examples,
SQL for real deployments), so a behavioural difference between them is a bug
waiting to surface only in production. Every test here is parametrised over both.
"""

from __future__ import annotations

import pytest
from cogria_agent import InMemoryConversationBackend
from cogria_agent.protocols import TITLE_MAX_CHARS, derive_title

pytest_plugins: list[str] = []

sqlalchemy = pytest.importorskip("sqlalchemy", reason="needs the sql extra")


@pytest.fixture(params=["memory", "sql"])
async def backend(request, tmp_path):
    if request.param == "memory":
        yield InMemoryConversationBackend()
        return
    from cogria_agent import SqlConversationBackend

    b = SqlConversationBackend(f"sqlite+aiosqlite:///{tmp_path}/t.db")
    await b.create_all()
    try:
        yield b
    finally:
        await b.dispose()


async def _seed(backend, *, user_id, first="hello there"):
    return await backend.create_conversation(
        first_message=first, model="m", user_id=user_id
    )


async def test_public_id_is_opaque_not_a_counter(backend):
    """Ids must not be guessable or reveal how many conversations exist."""
    a = await _seed(backend, user_id="u1")
    b = await _seed(backend, user_id="u1")
    assert a != b
    for cid in (a, b):
        assert isinstance(cid, str)
        assert len(cid) == 36 and cid.count("-") == 4  # uuid4 shape
        assert not cid.isdigit()


async def test_meta_reports_the_owner(backend):
    cid = await _seed(backend, user_id="u1")
    meta = await backend.fetch_meta(cid)
    assert meta["id"] == cid
    assert meta["user_id"] == "u1"
    assert meta["deleted_at"] is None


async def test_unknown_id_has_no_meta(backend):
    assert await backend.fetch_meta("00000000-0000-4000-8000-000000000000") == {}


async def test_list_is_scoped_to_the_owner(backend):
    mine = await _seed(backend, user_id="u1", first="mine")
    await _seed(backend, user_id="u2", first="theirs")

    rows = await backend.list_conversations(user_id="u1")
    assert [r["id"] for r in rows] == [mine]
    assert rows[0]["title"] == "mine"

    rows2 = await backend.list_conversations(user_id="u2")
    assert [r["title"] for r in rows2] == ["theirs"]


async def test_list_is_newest_first(backend):
    first = await _seed(backend, user_id="u1", first="first")
    second = await _seed(backend, user_id="u1", first="second")
    rows = await backend.list_conversations(user_id="u1")
    assert [r["id"] for r in rows] == [second, first]


async def test_list_paginates(backend):
    ids = [await _seed(backend, user_id="u1", first=f"c{i}") for i in range(5)]
    page = await backend.list_conversations(user_id="u1", limit=2, offset=1)
    # newest-first is ids reversed; offset 1 skips the newest
    assert [r["id"] for r in page] == list(reversed(ids))[1:3]


async def test_list_counts_messages(backend):
    cid = await _seed(backend, user_id="u1")
    await backend.append_messages(
        cid,
        messages=[{"role": "assistant", "content": {"text": "hi"}}],
        usage=None,
        model=None,
    )
    rows = await backend.list_conversations(user_id="u1")
    assert rows[0]["message_count"] == 2


async def test_soft_delete_hides_from_the_list_but_keeps_the_data(backend):
    cid = await _seed(backend, user_id="u1")
    assert await backend.soft_delete_conversation(cid) is True

    assert await backend.list_conversations(user_id="u1") == []
    # The row and its messages survive — only the listing filters it.
    meta = await backend.fetch_meta(cid)
    assert meta["deleted_at"] is not None
    assert len(await backend.fetch_messages_full(cid)) == 1


async def test_soft_delete_is_idempotent(backend):
    cid = await _seed(backend, user_id="u1")
    await backend.soft_delete_conversation(cid)
    first_stamp = (await backend.fetch_meta(cid))["deleted_at"]
    await backend.soft_delete_conversation(cid)
    assert (await backend.fetch_meta(cid))["deleted_at"] == first_stamp


async def test_appending_to_an_unknown_conversation_raises(backend):
    with pytest.raises(KeyError):
        await backend.append_messages(
            "00000000-0000-4000-8000-000000000000",
            messages=[{"role": "user", "content": {"text": "x"}}],
            usage=None,
            model=None,
        )


async def test_history_of_an_unknown_conversation_is_empty(backend):
    assert await backend.fetch_history("00000000-0000-4000-8000-000000000000") == []


async def test_users_without_identity_are_kept_separate_from_owned_rows(backend):
    """A project may run without user identity; those rows must not leak into a
    signed-in user's history, nor the reverse."""
    anon = await _seed(backend, user_id=None, first="anon")
    owned = await _seed(backend, user_id="u1", first="owned")

    assert [r["id"] for r in await backend.list_conversations(user_id=None)] == [anon]
    assert [r["id"] for r in await backend.list_conversations(user_id="u1")] == [owned]


# --- title derivation ------------------------------------------------------


def test_derive_title_collapses_whitespace():
    assert derive_title("  add\n a   todo  ") == "add a todo"


def test_derive_title_elides_long_messages():
    long = "x" * (TITLE_MAX_CHARS + 40)
    out = derive_title(long)
    assert out.endswith("…")
    assert len(out) == TITLE_MAX_CHARS + 1


def test_derive_title_of_nothing_is_empty():
    assert derive_title(None) == ""
    assert derive_title("   ") == ""
