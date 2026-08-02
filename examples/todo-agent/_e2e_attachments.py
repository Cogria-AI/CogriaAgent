"""Manual real-LLM E2E for attachments (NOT collected by pytest).

Proves the claim the feature exists to make: a file the user uploads becomes
part of the prompt, and both the file and the message survive in the database.

Runs the kernel in-process (httpx ASGITransport — no bound port) and:
  1. uploads a generated PDF containing one distinctive fact,
  2. asks the model a question only that PDF can answer,
  3. re-opens the SQLite file and checks the message, the attachment row and
     the message_attachments link are all there,
  4. asks a follow-up in the same conversation to prove replay re-injects the
     document without a re-upload,
  5. (only if AGENT_VISION_MODEL is set) repeats with an image.

    export OPENAI_BASE_URL=https://your-gateway/v1
    export OPENAI_API_KEY=sk-...
    export AGENT_MODEL=deepseek-chat          # any tool-use-capable model
    # export AGENT_VISION_MODEL=gpt-4o-mini   # optional, enables the image leg
    uv run --extra sql --extra attachments python examples/todo-agent/_e2e_attachments.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL")):
    raise SystemExit("Set OPENAI_API_KEY and OPENAI_BASE_URL (OpenAI-compatible gateway) first.")

WORKDIR = Path(tempfile.mkdtemp(prefix="cogria-e2e-"))
os.environ.setdefault("JWT_SECRET", "e2e-test-secret-please-rotate-aaaaaaaaaaaa")
os.environ.setdefault("AGENT_MODEL", "deepseek-chat")
# Durable persistence + uploads, both switched on the way an integrator would.
os.environ["AGENT_DB_URL"] = f"sqlite+aiosqlite:///{WORKDIR / 'agent.db'}"
os.environ["AGENT_ATTACHMENTS_DIR"] = str(WORKDIR / "attachments")

import jwt  # noqa: E402
import pymupdf  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from cogria_agent.sqlbackend import SqlConversationBackend  # noqa: E402
from cogria_agent.sqlschema import attachments as attachments_table  # noqa: E402
from cogria_agent.sqlschema import message_attachments  # noqa: E402
from todo_agent.app import _config, app  # noqa: E402

FACT = "The Q3 refund rate was 4.7 percent"


def _mint_jwt() -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "iss": _config.auth.jwt_issuer,
            "sub": "e2e-user",
            "role": "owner",
            "locale": "en",
            "iat": now,
            "exp": now + 900,
        },
        _config.auth.jwt_secret,
        algorithm="HS256",
    )


def _make_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 144), "Quarterly Report")
    page.insert_text((72, 180), FACT + ".")
    page.insert_text((72, 216), "Prepared by the finance team.")
    data = doc.tobytes()
    doc.close()
    return data


def _make_png() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((40, 100), "PURPLE ELEPHANT", fontsize=28)
    pix = page.get_pixmap(dpi=96)
    data = pix.tobytes("png")
    doc.close()
    return data


async def _chat(client: AsyncClient, token: str, message: str, *, conversation_id=None, attachment_ids=None) -> dict:
    body: dict = {"message": message}
    if conversation_id is not None:
        body["conversation_id"] = conversation_id
    if attachment_ids:
        body["attachment_ids"] = attachment_ids

    out = {"text": "", "conversation_id": None, "error": None}
    async with client.stream(
        "POST", "/chat", json=body, headers={"Authorization": f"Bearer {token}"}
    ) as resp:
        if resp.status_code != 200:
            out["error"] = f"HTTP {resp.status_code}: {(await resp.aread()).decode()[:400]}"
            return out
        event = None
        async for raw in resp.aiter_lines():
            if raw.startswith("event: "):
                event = raw[7:].strip()
            elif raw.startswith("data: "):
                import json

                payload = json.loads(raw[6:])
                if event == "text":
                    out["text"] += payload.get("delta", "")
                elif event == "conversation":
                    out["conversation_id"] = payload.get("conversation_id")
                elif event == "error":
                    out["error"] = payload.get("message")
    return out


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")
    return ok


async def main() -> int:
    token = _mint_jwt()
    results: list[bool] = []

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://e2e") as client:
        health = (await client.get("/health")).json()
        results.append(check("attachments enabled", health.get("attachments") is True))

        print("\n[1] upload a PDF")
        up = await client.post(
            "/attachments",
            files={"file": ("quarterly.pdf", _make_pdf(), "application/pdf")},
            headers={"Authorization": f"Bearer {token}"},
        )
        if up.status_code != 200:
            print(f"  upload failed: {up.status_code} {up.text[:300]}")
            return 1
        record = up.json()
        print(f"  id={record['id']} status={record['status']} chars={record['extracted_chars']}")
        results.append(check("PDF text extracted", (record.get("extracted_chars") or 0) > 0))

        print("\n[2] ask the model about it")
        turn = await _chat(
            client, token, "What was the Q3 refund rate in the attached report? Answer briefly.",
            attachment_ids=[record["id"]],
        )
        print(f"  model: {turn['text'][:300]}")
        if turn["error"]:
            print(f"  error: {turn['error']}")
        results.append(check("answer contains the fact from the PDF", "4.7" in turn["text"]))
        cid = turn["conversation_id"]

        print("\n[3] follow-up in the same conversation (replay, no re-upload)")
        follow = await _chat(client, token, "Who prepared it?", conversation_id=cid)
        print(f"  model: {follow['text'][:300]}")
        results.append(
            check("replayed document still readable", "finance" in follow["text"].lower())
        )

        if os.environ.get("AGENT_VISION_MODEL"):
            print("\n[4] upload an image (vision model configured)")
            img = await client.post(
                "/attachments",
                files={"file": ("sign.png", _make_png(), "image/png")},
                headers={"Authorization": f"Bearer {token}"},
            )
            results.append(check("image accepted", img.status_code == 200, img.text[:200]))
            if img.status_code == 200:
                vision = await _chat(
                    client, token, "What words appear in this image?",
                    attachment_ids=[img.json()["id"]],
                )
                print(f"  model: {vision['text'][:300]}")
                results.append(check("model read the image", "elephant" in vision["text"].lower()))
        else:
            print("\n[4] skipped — set AGENT_VISION_MODEL to test image uploads")

    print("\n[5] verify the database (fresh connection, as if after a restart)")
    backend = SqlConversationBackend(os.environ["AGENT_DB_URL"])
    try:
        rows = await backend.fetch_history(cid, for_llm=False)
        user_row = next((r for r in rows if r["role"] == "user"), {})
        refs = (user_row.get("content") or {}).get("attachments") or []
        results.append(check("user message persisted with its attachment", bool(refs)))
        print(f"  message content: {user_row.get('content')}")

        async with backend.engine.connect() as conn:
            links = (await conn.execute(select(message_attachments))).all()
            stored = (await conn.execute(select(attachments_table))).all()
        results.append(check("message_attachments link written", len(links) >= 1))
        results.append(
            check(
                "attachment row keeps the extracted text",
                bool(stored) and bool(stored[0].extracted_text),
                f"{len(stored[0].extracted_text or '')} chars" if stored else "no rows",
            )
        )
        print(f"  blobs on disk: {sorted(p.name for p in Path(os.environ['AGENT_ATTACHMENTS_DIR']).rglob('*') if p.is_file())}")
    finally:
        await backend.dispose()

    print(f"\n{sum(results)}/{len(results)} checks passed   (workdir: {WORKDIR})")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
