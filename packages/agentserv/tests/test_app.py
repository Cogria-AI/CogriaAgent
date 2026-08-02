"""App construction + JWT auth wiring (no LLM needed)."""

from __future__ import annotations

import time

import jwt
from fastapi.testclient import TestClient

from cogria_agent import (
    AgentConfig,
    AuthConfig,
    InMemoryConversationBackend,
    StaticCatalogProvider,
    build_app,
)

SECRET = "test-secret-please-ignore"


class _NoopExecutor:
    async def invoke(self, *, name, args, query=None, context=None):
        return {"ok": True, "data": {}}


def _client() -> TestClient:
    config = AgentConfig(auth=AuthConfig(jwt_secret=SECRET, jwt_issuer="cogria-agent"))
    app = build_app(
        config,
        conversation_backend=InMemoryConversationBackend(),
        action_executor=_NoopExecutor(),
        catalog_provider=StaticCatalogProvider({"actions": []}),
    )
    return TestClient(app)


def _token(**claims) -> str:
    payload = {"iss": "cogria-agent", "sub": "u1", "exp": int(time.time()) + 300, **claims}
    return jwt.encode(payload, SECRET, algorithm="HS256")


def test_health_open():
    r = _client().get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_whoami_requires_bearer():
    assert _client().get("/debug/whoami").status_code == 401


def test_whoami_rejects_bad_token():
    r = _client().get("/debug/whoami", headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401


def test_whoami_accepts_valid_token():
    r = _client().get("/debug/whoami", headers={"Authorization": f"Bearer {_token(locale='zh-CN', role='owner')}"})
    assert r.status_code == 200
    body = r.json()
    assert body["sub"] == "u1" and body["locale"] == "zh-CN" and body["role"] == "owner"
    # single-tenant: no tenant claims leak through
    assert "tid" not in body and "tslug" not in body


def test_chat_requires_message():
    r = _client().post("/chat", headers={"Authorization": f"Bearer {_token()}"}, json={})
    assert r.status_code == 422
