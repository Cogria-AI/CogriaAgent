"""menu-agent acceptance: the restaurant capability declared via the backend-py
SDK, driven through RegistryExecutor (the embedded bridge into the kernel).

Mirrors examples/todo-agent/tests/test_sdk.py on purpose — same flow, different
domain — to make "swap the business, kernel unchanged" concrete.
"""

from __future__ import annotations

import pytest

from cogria_backend import ProposalTokenService, Registry, RegistryExecutor
from menu_agent.actions import all_actions
from menu_agent.store import MenuStore


def _executor():
    import menu_agent.actions as actions

    actions.STORE = MenuStore().seed()  # fresh, seeded menu per test
    reg = Registry(all_actions())
    return RegistryExecutor(reg, ProposalTokenService("test-secret")), reg


@pytest.mark.asyncio
async def test_catalog_shape():
    _, reg = _executor()
    cat = reg.catalog()["actions"]
    names = {a["name"] for a in cat}
    assert names == {"list_dishes", "create_dish", "update_price", "set_availability"}
    # writes advertise confirmation; the read does not.
    confirm = {a["name"]: a["requires_confirm"] for a in cat}
    assert confirm == {
        "list_dishes": False,
        "create_dish": True,
        "update_price": True,
        "set_availability": True,
    }
    # slugs are kebab-cased names (contract rule).
    slugs = {a["url_slug"] for a in cat}
    assert "create-dish" in slugs


@pytest.mark.asyncio
async def test_list_seeded_menu():
    ex, _ = _executor()
    res = await ex.invoke(name="list_dishes", args={})
    assert res["ok"] and len(res["data"]["dishes"]) == 3


@pytest.mark.asyncio
async def test_list_filtered_by_category():
    ex, _ = _executor()
    res = await ex.invoke(name="list_dishes", args={"category": "Desserts"})
    dishes = res["data"]["dishes"]
    assert [d["name"] for d in dishes] == ["Tiramisu"]


@pytest.mark.asyncio
async def test_create_dish_propose_confirm():
    ex, _ = _executor()
    prop = await ex.invoke(
        name="create_dish",
        args={"name": "Espresso", "price": 3.0, "category": "Drinks"},
        query={"dry_run": "1"},
    )
    assert prop["data"]["requires_confirm"] is True
    token = prop["data"]["proposal_token"]

    # not created until confirmed
    before = await ex.invoke(name="list_dishes", args={"category": "Drinks"})
    assert before["data"]["dishes"] == []

    done = await ex.invoke(
        name="create_dish",
        args={"name": "Espresso", "price": 3.0, "category": "Drinks", "proposal_token": token},
    )
    assert done["ok"] and done["audit_id"].startswith("act_")
    after = await ex.invoke(name="list_dishes", args={"category": "Drinks"})
    assert [d["name"] for d in after["data"]["dishes"]] == ["Espresso"]


@pytest.mark.asyncio
async def test_create_dish_confirm_requires_token():
    ex, _ = _executor()
    res = await ex.invoke(
        name="create_dish", args={"name": "Soup", "price": 5.0, "category": "Starters"}
    )
    assert res["ok"] is False and res["error"]["code"] == "PROPOSAL_REQUIRED"


@pytest.mark.asyncio
async def test_update_price_records_before_after_diff():
    ex, _ = _executor()
    prop = await ex.invoke(
        name="update_price", args={"id": 1, "price": 14.0}, query={"dry_run": "1"}
    )
    token = prop["data"]["proposal_token"]
    done = await ex.invoke(name="update_price", args={"id": 1, "price": 14.0, "proposal_token": token})
    assert done["ok"]
    res = await ex.invoke(name="list_dishes", args={})
    dish = next(d for d in res["data"]["dishes"] if d["id"] == 1)
    assert dish["price"] == 14.0


@pytest.mark.asyncio
async def test_update_price_unknown_dish_rejected_at_propose():
    ex, _ = _executor()
    res = await ex.invoke(name="update_price", args={"id": 999, "price": 1.0}, query={"dry_run": "1"})
    assert res["ok"] is False and res["error"]["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_set_availability_round_trip():
    ex, _ = _executor()
    prop = await ex.invoke(
        name="set_availability", args={"id": 1, "available": False}, query={"dry_run": "1"}
    )
    token = prop["data"]["proposal_token"]
    done = await ex.invoke(
        name="set_availability", args={"id": 1, "available": False, "proposal_token": token}
    )
    assert done["ok"]
    res = await ex.invoke(name="list_dishes", args={"available_only": True})
    assert all(d["id"] != 1 for d in res["data"]["dishes"])


def test_app_builds_with_kernel():
    """The restaurant project wires SDK actions into the kernel without touching it."""
    from menu_agent.app import app

    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/chat" in paths and "/agent-auth/exchange" in paths
