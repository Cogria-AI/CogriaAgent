"""Smoke tests for the scaffolded actions — proves your project runs before you
wire in a real LLM. Driven through the same RegistryExecutor the kernel uses.
"""

from __future__ import annotations

import pytest

from app.actions import AddTask, ListTasks, SetPriority
from app.store import TaskStore
from cogria_backend import ProposalTokenService, Registry, RegistryExecutor


def _executor():
    import app.actions as actions

    actions.STORE = TaskStore()  # fresh store per test
    reg = Registry([ListTasks(), AddTask(), SetPriority()])
    return RegistryExecutor(reg, ProposalTokenService("test-secret")), reg


@pytest.mark.asyncio
async def test_catalog():
    _, reg = _executor()
    names = {a["name"] for a in reg.catalog()["actions"]}
    assert names == {"list_tasks", "add_task", "set_priority"}


@pytest.mark.asyncio
async def test_add_task_propose_confirm():
    ex, _ = _executor()
    assert (await ex.invoke(name="list_tasks", args={}))["data"]["tasks"] == []

    prop = await ex.invoke(name="add_task", args={"title": "ship demo"}, query={"dry_run": "1"})
    token = prop["data"]["proposal_token"]
    assert (await ex.invoke(name="list_tasks", args={}))["data"]["tasks"] == []  # not yet

    done = await ex.invoke(name="add_task", args={"title": "ship demo", "proposal_token": token})
    assert done["ok"] and done["audit_id"].startswith("act_")
    tasks = (await ex.invoke(name="list_tasks", args={}))["data"]["tasks"]
    assert [t["title"] for t in tasks] == ["ship demo"]


@pytest.mark.asyncio
async def test_set_priority_validates_target():
    ex, _ = _executor()
    miss = await ex.invoke(name="set_priority", args={"id": 99, "priority": "high"}, query={"dry_run": "1"})
    assert miss["ok"] is False and miss["error"]["code"] == "NOT_FOUND"


def test_app_builds_with_kernel():
    from app.app import app

    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/chat" in paths and "/agent-auth/exchange" in paths
