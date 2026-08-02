"""Todo executor: read directly, writes go through propose/confirm."""

from __future__ import annotations

import pytest

from todo_agent.actions import TodoExecutor


@pytest.mark.asyncio
async def test_list_is_direct():
    ex = TodoExecutor()
    out = await ex.invoke(name="list_todos", args={})
    assert out["ok"] and out["data"]["todos"] == []


@pytest.mark.asyncio
async def test_add_propose_then_confirm():
    ex = TodoExecutor()
    # propose (dry_run): no mutation, returns a token + summary
    proposal = await ex.invoke(name="add_todo", args={"title": "buy milk"}, query={"dry_run": "1"})
    assert proposal["data"]["requires_confirm"] is True
    token = proposal["data"]["proposal_token"]
    assert (await ex.invoke(name="list_todos", args={}))["data"]["todos"] == []  # not yet added

    # confirm with the token: executes
    done = await ex.invoke(name="add_todo", args={"title": "buy milk", "proposal_token": token})
    assert done["ok"] and done["data"]["title"] == "buy milk"
    todos = (await ex.invoke(name="list_todos", args={}))["data"]["todos"]
    assert len(todos) == 1 and todos[0]["title"] == "buy milk"


@pytest.mark.asyncio
async def test_confirm_without_token_refused():
    ex = TodoExecutor()
    out = await ex.invoke(name="add_todo", args={"title": "x"})
    assert out["ok"] is False and out["error"]["code"] == "PROPOSAL_REQUIRED"


@pytest.mark.asyncio
async def test_token_is_one_shot():
    ex = TodoExecutor()
    token = (await ex.invoke(name="add_todo", args={"title": "x"}, query={"dry_run": "1"}))["data"]["proposal_token"]
    await ex.invoke(name="add_todo", args={"title": "x", "proposal_token": token})
    reuse = await ex.invoke(name="add_todo", args={"title": "x", "proposal_token": token})
    assert reuse["ok"] is False and reuse["error"]["code"] == "PROPOSAL_EXPIRED"


@pytest.mark.asyncio
async def test_complete_flow():
    ex = TodoExecutor()
    tok = (await ex.invoke(name="add_todo", args={"title": "task"}, query={"dry_run": "1"}))["data"]["proposal_token"]
    await ex.invoke(name="add_todo", args={"title": "task", "proposal_token": tok})
    ctok = (await ex.invoke(name="complete_todo", args={"id": 1}, query={"dry_run": "1"}))["data"]["proposal_token"]
    out = await ex.invoke(name="complete_todo", args={"id": 1, "proposal_token": ctok})
    assert out["ok"] and out["data"]["done"] is True


def test_app_builds():
    # Importing builds the FastAPI app with the kernel + todo seams (no LLM call).
    from todo_agent.app import app

    assert any(r.path == "/chat" for r in app.routes)
