"""DEV ONLY — stand in for the restaurant backend's /agent-auth/exchange.

A real deployment mints the JWT from the logged-in user's session. For the local
demo we mint one unconditionally so the BFF has an exchange endpoint to call.
NEVER expose this in production.
"""

from __future__ import annotations

import time

import jwt
from fastapi import FastAPI, Request


def add_dev_exchange(app: FastAPI, config) -> None:
    @app.post("/agent-auth/exchange")
    async def dev_exchange(request: Request) -> dict:
        try:
            body = await request.json()
        except Exception:
            body = {}
        now = int(time.time())
        ttl = 1800
        token = jwt.encode(
            {
                "iss": config.auth.jwt_issuer,
                "sub": "demo-user",
                "role": "owner",
                "locale": body.get("locale") or config.locales.default,
                "iat": now,
                "exp": now + ttl,
            },
            config.auth.jwt_secret,
            algorithm="HS256",
        )
        return {"token": token, "expires_at": now + ttl, "user": {"role": "owner"}}
