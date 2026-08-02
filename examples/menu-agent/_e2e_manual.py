"""Manual real-LLM E2E for the menu-agent example (NOT collected by pytest).

Drives the kernel's /chat SSE endpoint in-process (httpx ASGITransport — no bound
port, no long-running server) through three turns: read, propose, confirm. Proves
the full path LLM ↔ kernel ↔ backend-py SDK ↔ actions works against a real model.

Point it at any OpenAI-compatible gateway via env vars:

    export OPENAI_BASE_URL=https://your-gateway/v1
    export OPENAI_API_KEY=sk-...
    export AGENT_MODEL=deepseek-chat        # any tool-use-capable model
    uv run python examples/menu-agent/_e2e_manual.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time

if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL")):
    raise SystemExit("Set OPENAI_API_KEY and OPENAI_BASE_URL (OpenAI-compatible gateway) first.")
os.environ.setdefault("JWT_SECRET", "e2e-test-secret-please-rotate-aaaaaaaaaaaa")
os.environ.setdefault("AGENT_MODEL", "deepseek-chat")

import jwt  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from menu_agent.app import _config, app  # noqa: E402


def _mint_jwt() -> str:
    now = int(time.time())
    return jwt.encode(
        {"iss": _config.auth.jwt_issuer, "sub": "e2e-user", "role": "owner",
         "locale": "en", "iat": now, "exp": now + 600},
        _config.auth.jwt_secret, algorithm="HS256",
    )


async def _chat(client: AsyncClient, token: str, message: str, *,
                conversation_id=None, proposal_token=None) -> dict:
    """POST /chat, parse the SSE stream, return collected events."""
    body: dict = {"message": message}
    if conversation_id is not None:
        body["conversation_id"] = conversation_id
    if proposal_token is not None:
        body["proposal_token"] = proposal_token

    collected = {"text": "", "tool_calls": [], "confirm": None, "conversation_id": None, "error": None}
    async with client.stream("POST", "/chat", json=body,
                             headers={"Authorization": f"Bearer {token}"}) as resp:
        if resp.status_code != 200:
            collected["error"] = f"HTTP {resp.status_code}: {await resp.aread()}"
            return collected
        event = None
        async for raw in resp.aiter_lines():
            if raw.startswith("event: "):
                event = raw[7:].strip()
            elif raw.startswith("data: "):
                data = json.loads(raw[6:])
                if event == "text":
                    collected["text"] += data.get("delta", "")
                elif event == "tool_call":
                    collected["tool_calls"].append(data["name"])
                elif event == "confirm_required":
                    collected["confirm"] = data
                elif event == "conversation":
                    collected["conversation_id"] = data.get("conversation_id")
                elif event == "error":
                    collected["error"] = data
    return collected


def _ok(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  {'✅' if cond else '❌'} {label}" + (f" — {detail}" if detail else ""))
    return cond


async def main() -> int:
    print(f"Model: {os.environ['AGENT_MODEL']} @ {os.environ['OPENAI_BASE_URL']}\n")
    transport = ASGITransport(app=app)
    token = _mint_jwt()
    passed = True

    async with AsyncClient(transport=transport, base_url="http://e2e") as client:
        # --- Turn 1: READ ----------------------------------------------------
        print("Turn 1 — read: \"What's on the menu?\"")
        r1 = await _chat(client, token, "What's on the menu? Just list the dish names.")
        conv = r1["conversation_id"]
        passed &= _ok("no error", r1["error"] is None, str(r1["error"] or ""))
        passed &= _ok("called list_dishes", "list_dishes" in r1["tool_calls"], str(r1["tool_calls"]))
        passed &= _ok("answered with a dish", any(
            d in r1["text"] for d in ("Margherita", "Caesar", "Tiramisu")), repr(r1["text"][:120]))

        # --- Turn 2: PROPOSE -------------------------------------------------
        print("\nTurn 2 — propose: \"Add Espresso, 3.00, Drinks\"")
        r2 = await _chat(client, token,
                         "Add a new dish: Espresso, price 3.00, category Drinks.",
                         conversation_id=conv)
        passed &= _ok("no error", r2["error"] is None, str(r2["error"] or ""))
        passed &= _ok("called create_dish (dry_run)", "create_dish" in r2["tool_calls"], str(r2["tool_calls"]))
        got_token = bool(r2["confirm"] and r2["confirm"].get("proposal_token"))
        passed &= _ok("emitted confirm_required + token", got_token,
                      (r2["confirm"] or {}).get("summary", ""))

        # --- Turn 3: CONFIRM -------------------------------------------------
        if got_token:
            ptok = r2["confirm"]["proposal_token"]
            print("\nTurn 3 — confirm: [CONFIRMED]")
            r3 = await _chat(client, token, "[CONFIRMED]", conversation_id=conv, proposal_token=ptok)
            passed &= _ok("no error", r3["error"] is None, str(r3["error"] or ""))
            passed &= _ok("re-called create_dish with token", "create_dish" in r3["tool_calls"], str(r3["tool_calls"]))
            # Ground truth: the dish is actually in the store now.
            from menu_agent.actions import STORE
            names = [d.name for d in STORE.items.values()]
            passed &= _ok("Espresso persisted in store", any("Espresso" in n for n in names), str(names))

    print("\n" + ("🎉 E2E PASSED" if passed else "💥 E2E FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
