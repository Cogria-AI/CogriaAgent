"""Attachments on the durable path: metadata rows, ownership, message links.

The behaviour has to match InMemoryAttachmentRepository exactly (the kernel is
written against the protocol), plus the thing only the SQL pair can do: keep a
message and the files it carried joined up in one database.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy", reason="needs the `sql` extra")
pytest.importorskip("aiosqlite", reason="needs the `sql` extra")

from cogria_agent.attachments import (  # noqa: E402
    InMemoryAttachmentRepository,
    SqlAttachmentRepository,
)
from cogria_agent.sqlbackend import SqlConversationBackend  # noqa: E402
from cogria_agent.sqlschema import attachments as attachments_table  # noqa: E402
from cogria_agent.sqlschema import conversations, message_attachments  # noqa: E402
from sqlalchemy import select  # noqa: E402

FIELDS = dict(
    filename="menu.pdf",
    mime="application/pdf",
    size_bytes=1234,
    sha256="d" * 64,
    storage_key=f"dd/dd/{'d' * 64}",
    kind="document",
    status="extracting",
)


@pytest.fixture
async def sql_pair(tmp_path):
    backend = SqlConversationBackend(f"sqlite+aiosqlite:///{tmp_path / 'agent.db'}")
    repo = SqlAttachmentRepository(engine=backend.engine)
    await backend.create_all()
    try:
        yield backend, repo
    finally:
        await backend.dispose()


@pytest.fixture(params=["inmemory", "sql"])
async def repo(request, sql_pair):
    yield InMemoryAttachmentRepository() if request.param == "inmemory" else sql_pair[1]


async def test_create_extract_and_read_back(repo):
    created = await repo.create(id="att_1", owner_sub="u1", **FIELDS)
    assert created["status"] == "extracting"

    await repo.mark_ready("att_1", kind="text", page_count=7, extracted_text="hello world")
    record = await repo.get("att_1", owner_sub="u1")
    assert record["status"] == "ready"
    assert record["page_count"] == 7
    assert record["extracted_text"] == "hello world"
    assert record["extracted_chars"] == 11


async def test_reads_are_scoped_to_the_owner(repo):
    await repo.create(id="att_1", owner_sub="u1", **FIELDS)
    assert await repo.get("att_1", owner_sub="u2") is None
    assert await repo.get_many(["att_1"], owner_sub="u2") == []
    assert await repo.delete("att_1", owner_sub="u2") is None
    assert await repo.get("att_1", owner_sub="u1") is not None


async def test_get_many_keeps_request_order(repo):
    for i in (1, 2, 3):
        await repo.create(id=f"att_{i}", owner_sub="u1", **{**FIELDS, "sha256": str(i) * 64})
    got = await repo.get_many(["att_3", "att_1"], owner_sub="u1")
    assert [r["id"] for r in got] == ["att_3", "att_1"]


async def test_find_by_sha256_only_matches_ready_and_owned(repo):
    await repo.create(id="att_1", owner_sub="u1", **FIELDS)
    # Still extracting -> not a dedup candidate yet.
    assert await repo.find_by_sha256(FIELDS["sha256"], owner_sub="u1") is None
    await repo.mark_ready("att_1", kind="text", page_count=None, extracted_text="x")
    assert (await repo.find_by_sha256(FIELDS["sha256"], owner_sub="u1"))["id"] == "att_1"
    assert await repo.find_by_sha256(FIELDS["sha256"], owner_sub="u2") is None


async def test_count_by_storage_key_sees_every_owner(repo):
    """Ref-count for the shared-blob delete guard — deliberately NOT scoped to
    an owner, since the blob is shared across them."""
    await repo.create(id="att_1", owner_sub="u1", **FIELDS)
    await repo.create(id="att_2", owner_sub="u2", **FIELDS)
    assert await repo.count_by_storage_key(FIELDS["storage_key"]) == 2

    await repo.delete("att_1", owner_sub="u1")
    assert await repo.count_by_storage_key(FIELDS["storage_key"]) == 1
    assert await repo.count_by_storage_key("no/su/ch") == 0


async def test_mark_failed_records_the_reason(repo):
    await repo.create(id="att_1", owner_sub="u1", **FIELDS)
    await repo.mark_failed("att_1", error="No extractable text.")
    record = await repo.get("att_1", owner_sub="u1")
    assert record["status"] == "failed" and "No extractable" in record["error"]


async def test_message_links_and_conversation_stamp(sql_pair):
    backend, repo = sql_pair
    await repo.create(id="att_1", owner_sub="u1", **FIELDS)
    await repo.mark_ready("att_1", kind="text", page_count=1, extracted_text="body")

    content = {"text": "read this", "attachments": [{"id": "att_1", "name": "menu.pdf"}]}
    cid = await backend.create_conversation(
        first_message="read this", model="m", first_content=content
    )

    async with backend.engine.connect() as conn:
        links = (await conn.execute(select(message_attachments))).all()
        stamped = await conn.scalar(
            select(attachments_table.c.conversation_id).where(attachments_table.c.id == "att_1")
        )
    assert len(links) == 1 and links[0].attachment_id == "att_1" and links[0].ordinal == 0
    # The attachment row is stamped with the INTERNAL conversation key (the FK
    # target), not the public id the API hands out.
    async with backend.engine.connect() as conn:
        internal = await conn.scalar(
            select(conversations.c.id).where(conversations.c.public_id == cid)
        )
    assert stamped == internal

    # The JSON copy is what replay reads — no join required.
    rows = await backend.fetch_history(cid)
    assert rows[0]["content"]["attachments"][0]["id"] == "att_1"


async def test_unknown_attachment_ids_do_not_break_the_write(sql_pair):
    """A stale id in a message must never cost the user their message."""
    backend, _ = sql_pair
    content = {"text": "hi", "attachments": [{"id": "att_ghost"}]}
    cid = await backend.create_conversation(first_message="hi", model="m", first_content=content)
    async with backend.engine.connect() as conn:
        assert (await conn.execute(select(message_attachments))).all() == []
    assert (await backend.fetch_history(cid))[0]["content"]["attachments"][0]["id"] == "att_ghost"


async def test_deleting_an_attachment_drops_its_links(sql_pair):
    backend, repo = sql_pair
    await repo.create(id="att_1", owner_sub="u1", **FIELDS)
    await repo.mark_ready("att_1", kind="text", page_count=None, extracted_text="body")
    await backend.create_conversation(
        first_message="x", model="m", first_content={"text": "x", "attachments": [{"id": "att_1"}]}
    )

    assert await repo.delete("att_1", owner_sub="u1") is not None
    async with backend.engine.connect() as conn:
        assert (await conn.execute(select(message_attachments))).all() == []
