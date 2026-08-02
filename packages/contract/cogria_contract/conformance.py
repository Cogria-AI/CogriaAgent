"""Contract conformance suite — point it at any backend's base URL and it
verifies the wire contract over plain HTTP. SDK-independent: a hand-written
Node/Go/PHP backend can self-check without importing anything Python-specific.

Usage:
    from cogria_contract.conformance import run_conformance
    checks = await run_conformance(
        "http://127.0.0.1:8000",
        read={"name": "list_todos", "args": {}},
        write={"name": "add_todo", "args": {"title": "x"}, "tampered_args": {"title": "y"}},
        headers={"Authorization": "Bearer ..."},
    )
    assert all(c.passed for c in checks)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .types import PROPOSAL_EXPIRED, PROPOSAL_REQUIRED, url_slug_for


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def _action_url(base: str, name: str) -> str:
    return f"{base.rstrip('/')}/agent-actions/{url_slug_for(name)}"


async def run_conformance(
    base_url: str,
    *,
    read: dict[str, Any] | None = None,
    write: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[Check]:
    """Run the full suite. `read`/`write` describe a sample action of each kind.
    `write` may include `tampered_args` to verify fingerprint binding.

    Pass `client` to reuse a configured AsyncClient (e.g. an ASGITransport one
    for in-process tests); otherwise a default client is created."""
    checks: list[Check] = []
    if client is not None:
        await _run(client, base_url, read, write, checks)
        return checks
    h = {"Accept": "application/json", **(headers or {})}
    async with httpx.AsyncClient(timeout=30, headers=h) as c:
        await _run(c, base_url, read, write, checks)
    return checks


async def _run(
    client: httpx.AsyncClient,
    base_url: str,
    read: dict[str, Any] | None,
    write: dict[str, Any] | None,
    checks: list[Check],
) -> None:
    await _check_catalog(client, base_url, checks)
    if read:
        await _check_read(client, base_url, read, checks)
    if write:
        await _check_propose_confirm(client, base_url, write, checks)


async def _check_catalog(client: httpx.AsyncClient, base: str, out: list[Check]) -> None:
    try:
        r = await client.get(f"{base.rstrip('/')}/agent-actions/_catalog")
        body = r.json()
        actions = body.get("actions")
        ok = r.status_code == 200 and isinstance(actions, list)
        # Each action must carry the LLM-routing essentials.
        shape_ok = all(
            isinstance(a, dict) and {"name", "description", "params_schema"} <= set(a) for a in (actions or [])
        )
        out.append(Check("catalog.shape", bool(ok and shape_ok), f"status={r.status_code} actions={len(actions or [])}"))
        # url_slug, if present, must be the mechanical mapping.
        slug_ok = all(a.get("url_slug", url_slug_for(a["name"])) == url_slug_for(a["name"]) for a in (actions or []))
        out.append(Check("catalog.url_slug_mapping", slug_ok))
    except Exception as e:  # noqa: BLE001
        out.append(Check("catalog.shape", False, f"{type(e).__name__}: {e}"))


async def _check_read(client: httpx.AsyncClient, base: str, read: dict[str, Any], out: list[Check]) -> None:
    try:
        r = await client.post(_action_url(base, read["name"]), json=read.get("args", {}))
        env = r.json()
        out.append(Check("read.envelope_ok", env.get("ok") is True, str(env)[:160]))
    except Exception as e:  # noqa: BLE001
        out.append(Check("read.envelope_ok", False, f"{type(e).__name__}: {e}"))


async def _check_propose_confirm(client: httpx.AsyncClient, base: str, write: dict[str, Any], out: list[Check]) -> None:
    name = write["name"]
    args = write.get("args", {})
    url = _action_url(base, name)

    # 1) propose (dry_run): returns a proposal_token, does NOT execute.
    token = None
    try:
        r = await client.post(url, params={"dry_run": "1"}, json=args)
        env = r.json()
        data = env.get("data") or {}
        token = data.get("proposal_token")
        out.append(
            Check(
                "write.propose_returns_token",
                bool(env.get("ok") and data.get("requires_confirm") and token),
                str(env)[:160],
            )
        )
    except Exception as e:  # noqa: BLE001
        out.append(Check("write.propose_returns_token", False, f"{type(e).__name__}: {e}"))

    # 2) confirm without a token -> PROPOSAL_REQUIRED.
    try:
        r = await client.post(url, json=args)
        env = r.json()
        code = (env.get("error") or {}).get("code")
        out.append(Check("write.no_token_refused", env.get("ok") is False and code == PROPOSAL_REQUIRED, str(env)[:160]))
    except Exception as e:  # noqa: BLE001
        out.append(Check("write.no_token_refused", False, f"{type(e).__name__}: {e}"))

    # 3) tampered params with the token -> rejected (fingerprint binding).
    if token and "tampered_args" in write:
        try:
            r = await client.post(url, json={**write["tampered_args"], "proposal_token": token})
            env = r.json()
            out.append(Check("write.fingerprint_binding", env.get("ok") is False, str(env)[:160]))
        except Exception as e:  # noqa: BLE001
            out.append(Check("write.fingerprint_binding", False, f"{type(e).__name__}: {e}"))

    # 4) confirm with the token -> executes (ok).
    if token:
        try:
            r = await client.post(url, json={**args, "proposal_token": token})
            env = r.json()
            out.append(Check("write.confirm_executes", env.get("ok") is True, str(env)[:160]))
        except Exception as e:  # noqa: BLE001
            out.append(Check("write.confirm_executes", False, f"{type(e).__name__}: {e}"))

        # 5) reuse the same token -> one-shot, PROPOSAL_EXPIRED.
        try:
            r = await client.post(url, json={**args, "proposal_token": token})
            env = r.json()
            code = (env.get("error") or {}).get("code")
            out.append(Check("write.token_one_shot", env.get("ok") is False and code == PROPOSAL_EXPIRED, str(env)[:160]))
        except Exception as e:  # noqa: BLE001
            out.append(Check("write.token_one_shot", False, f"{type(e).__name__}: {e}"))


def format_report(checks: list[Check]) -> str:
    lines = [f"{'PASS' if c.passed else 'FAIL'}  {c.name}" + (f"  — {c.detail}" if not c.passed and c.detail else "") for c in checks]
    passed = sum(c.passed for c in checks)
    lines.append(f"\n{passed}/{len(checks)} checks passed")
    return "\n".join(lines)
