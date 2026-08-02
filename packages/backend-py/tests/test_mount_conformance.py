"""The FastAPI mount satisfies the SDK-independent contract conformance suite."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from cogria_backend import ProposalTokenService, Registry
from cogria_backend.fastapi_mount import mount_agent_actions
from cogria_contract.conformance import format_report, run_conformance

from sample_actions import CreateThing, ListThings, reset


@pytest.mark.asyncio
async def test_mount_passes_conformance():
    reset()
    app = FastAPI()
    mount_agent_actions(app, Registry([ListThings(), CreateThing()]), ProposalTokenService("test-secret"))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        checks = await run_conformance(
            "http://test",
            read={"name": "list_things", "args": {}},
            write={"name": "create_thing", "args": {"label": "x"}, "tampered_args": {"label": "y"}},
            client=client,
        )
    assert checks, "no checks ran"
    assert all(c.passed for c in checks), "\n" + format_report(checks)
