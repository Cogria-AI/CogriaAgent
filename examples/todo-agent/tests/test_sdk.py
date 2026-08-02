"""C1 acceptance: the todo capability declared via the backend-py SDK, driven
through RegistryExecutor (the embedded bridge into the agentserv kernel)."""

from __future__ import annotations

import pytest

from cogria_backend import ProposalTokenService, Registry, RegistryExecutor
from todo_agent.sdk_actions import AddTodo, CompleteTodo, ListTodos, STORE


def _executor():
    STORE.items.clear()
    STORE._next = 1
    reg = Registry([ListTodos(), AddTodo(), CompleteTodo()])
    return RegistryExecutor(reg, ProposalTokenService("test-secret")), reg


@pytest.mark.asyncio
async def test_sdk_catalog_matches_handwritten():
    _, reg = _executor()
    names = {a["name"] for a in reg.catalog()["actions"]}
    assert names == {"list_todos", "add_todo", "complete_todo"}


@pytest.mark.asyncio
async def test_sdk_read_propose_confirm_verify():
    ex, _ = _executor()
    assert (await ex.invoke(name="list_todos", args={}))["data"]["todos"] == []

    prop = await ex.invoke(name="add_todo", args={"title": "milk"}, query={"dry_run": "1"})
    token = prop["data"]["proposal_token"]
    assert (await ex.invoke(name="list_todos", args={}))["data"]["todos"] == []  # not yet

    done = await ex.invoke(name="add_todo", args={"title": "milk", "proposal_token": token})
    assert done["ok"] and done["audit_id"].startswith("act_")
    todos = (await ex.invoke(name="list_todos", args={}))["data"]["todos"]
    assert [t["title"] for t in todos] == ["milk"]


@pytest.mark.asyncio
async def test_sdk_complete_validate_target():
    ex, _ = _executor()
    miss = await ex.invoke(name="complete_todo", args={"id": 999}, query={"dry_run": "1"})
    assert miss["ok"] is False and miss["error"]["code"] == "NOT_FOUND"


def test_app_sdk_builds_with_kernel():
    # A Python project wires SDK actions into the kernel without touching it.
    from todo_agent.app_sdk import app

    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/chat" in paths and "/agent-auth/exchange" in paths
