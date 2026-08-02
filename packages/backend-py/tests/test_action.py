"""RegistryExecutor: read direct, write through propose/confirm + token rules."""

from __future__ import annotations

import pytest

from cogria_backend import ProposalTokenService, Registry, RegistryExecutor

from sample_actions import CreateThing, ListThings, reset


def _executor():
    reset()
    reg = Registry([ListThings(), CreateThing()])
    tokens = ProposalTokenService("test-secret")
    return RegistryExecutor(reg, tokens), reg


@pytest.mark.asyncio
async def test_catalog_shape():
    _, reg = _executor()
    cat = reg.catalog()
    names = {a["name"]: a for a in cat["actions"]}
    assert names["create_thing"]["requires_confirm"] is True
    assert names["create_thing"]["url_slug"] == "create-thing"
    assert names["list_things"]["requires_confirm"] is False


@pytest.mark.asyncio
async def test_read_direct():
    ex, _ = _executor()
    out = await ex.invoke(name="list_things", args={})
    assert out["ok"] and out["data"]["things"] == [] and "0 thing" in out["message"]


@pytest.mark.asyncio
async def test_write_propose_then_confirm():
    ex, _ = _executor()
    prop = await ex.invoke(name="create_thing", args={"label": "x"}, query={"dry_run": "1"})
    assert prop["data"]["requires_confirm"] and prop["data"]["proposal_token"]
    token = prop["data"]["proposal_token"]
    # not yet created
    assert (await ex.invoke(name="list_things", args={}))["data"]["things"] == []

    done = await ex.invoke(name="create_thing", args={"label": "x", "proposal_token": token})
    assert done["ok"] and done["data"]["label"] == "x" and done["audit_id"].startswith("act_")
    assert len((await ex.invoke(name="list_things", args={}))["data"]["things"]) == 1


@pytest.mark.asyncio
async def test_no_token_refused():
    ex, _ = _executor()
    out = await ex.invoke(name="create_thing", args={"label": "x"})
    assert out["ok"] is False and out["error"]["code"] == "PROPOSAL_REQUIRED"


@pytest.mark.asyncio
async def test_token_one_shot():
    ex, _ = _executor()
    token = (await ex.invoke(name="create_thing", args={"label": "x"}, query={"dry_run": "1"}))["data"]["proposal_token"]
    await ex.invoke(name="create_thing", args={"label": "x", "proposal_token": token})
    reuse = await ex.invoke(name="create_thing", args={"label": "x", "proposal_token": token})
    assert reuse["ok"] is False and reuse["error"]["code"] == "PROPOSAL_EXPIRED"


@pytest.mark.asyncio
async def test_fingerprint_binding():
    ex, _ = _executor()
    token = (await ex.invoke(name="create_thing", args={"label": "x"}, query={"dry_run": "1"}))["data"]["proposal_token"]
    # confirm with tampered params -> rejected
    out = await ex.invoke(name="create_thing", args={"label": "DIFFERENT", "proposal_token": token})
    assert out["ok"] is False and out["error"]["code"] == "PROPOSAL_EXPIRED"


@pytest.mark.asyncio
async def test_missing_required_field():
    ex, _ = _executor()
    out = await ex.invoke(name="create_thing", args={}, query={"dry_run": "1"})
    assert out["ok"] is False and out["error"]["code"] == "VALIDATION_FAILED"
