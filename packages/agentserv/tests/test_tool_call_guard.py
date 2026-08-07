"""Per-response tool-call guards: dedup, cap, confirmed-write protection, and
the provider's real finish_reason.

max_turns caps rounds, not how wide any single response is. Actions can cost
the caller real money (a paid external API, credits deducted per call), so a
response carrying the same call twenty times must not run it twenty times.
These tests pin that, plus the id-based result pairing that skipping any call
depends on.
"""

from __future__ import annotations

import json
import re
import time

import jwt
import pytest
from cogria_agent import (
    AgentConfig,
    AuthConfig,
    InMemoryConversationBackend,
    StaticCatalogProvider,
    build_app,
)
from cogria_agent.config import SummarizerConfig
from cogria_agent.graph import build_graph
from cogria_agent.tools import build_tools_from_catalog
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk

SECRET = "test-secret-please-ignore"

CATALOG = {
    "actions": [
        {
            "name": "search_records",
            "description": "Search records. Costs credits per call.",
            "params_schema": {
                "type": "object",
                "properties": {"topic": {"type": "string", "description": "what to find"}},
                "required": ["topic"],
            },
            "requires_confirm": False,
        },
        {
            "name": "update_record",
            "description": "Update a record. Write action.",
            "params_schema": {
                "type": "object",
                "properties": {"target": {"type": "string", "description": "record id"}},
                "required": ["target"],
            },
            "requires_confirm": True,
        },
    ]
}


class _RecordingExecutor:
    """Implements protocols.ActionExecutor; records every dispatch."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def invoke(self, *, name, args, query=None, context=None):
        self.calls.append({"name": name, "args": args, "query": query})
        return {"ok": True, "data": {"n": len(self.calls)}, "message": f"{name} ran"}


class _ScriptedLLM(BaseChatModel):
    """Stands in for ChatOpenAI: replays one prepared chunk per round.

    A real BaseChatModel, not a duck-typed stub — astream_events only emits the
    on_chat_model_* events the server reads finish_reason from when the thing
    streaming is an actual Runnable.
    """

    script: list

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        chunk = self.script.pop(0) if self.script else AIMessageChunk(content="done")
        yield ChatGenerationChunk(message=chunk)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("streaming only")


class _ScriptedFactory:
    def __init__(self, script: list[AIMessageChunk]) -> None:
        self._llm = _ScriptedLLM(script=script)

    def chat_llm(self, model=None):
        return self._llm

    def summary_llm(self):
        return self._llm

    def model_name(self):
        return "test-model"


class _Prompt:
    def system_prompt(self, *, locale=None) -> str:
        return "system"


def _chunk(calls: list[tuple[str, dict, str]], *, finish: str = "tool_calls") -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": name,
                "args": json.dumps(args),
                "id": cid,
                "index": i,
                "type": "tool_call_chunk",
            }
            for i, (name, args, cid) in enumerate(calls)
        ],
        response_metadata={"finish_reason": finish},
    )


async def _run(script: list[AIMessageChunk], *, max_tool_calls: int = 8):
    """Drive the graph once and hand back (tool messages, executor)."""
    executor = _RecordingExecutor()
    tools = build_tools_from_catalog(CATALOG, executor=executor)
    graph = build_graph(
        tools,
        llm_factory=_ScriptedFactory([*script, AIMessageChunk(content="all done")]),
        prompt_provider=_Prompt(),
        max_tool_calls=max_tool_calls,
    )
    state = await graph.ainvoke({"messages": [HumanMessage(content="hi")], "turns": 0})
    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    return tool_msgs, executor


def _envelope(msg: ToolMessage) -> dict:
    return json.loads(msg.content)


def _code(msg: ToolMessage) -> str | None:
    return _envelope(msg).get("error", {}).get("code")


# ---------------------------------------------------------------- dedup / cap


@pytest.mark.asyncio
async def test_identical_calls_run_once_and_share_the_result():
    """The charge happens once; the model still sees a result for every call."""
    calls = [("search_records", {"topic": "alpha"}, f"c{i}") for i in range(20)]
    tool_msgs, executor = await _run([_chunk(calls)])

    assert len(executor.calls) == 1
    # Every tool_call_id is answered — the API rejects a turn where one isn't.
    assert [m.tool_call_id for m in tool_msgs] == [f"c{i}" for i in range(20)]
    assert len({m.content for m in tool_msgs}) == 1
    assert _envelope(tool_msgs[0])["ok"] is True


@pytest.mark.asyncio
async def test_differing_args_are_not_treated_as_duplicates():
    calls = [
        ("search_records", {"topic": "alpha"}, "c1"),
        ("search_records", {"topic": "beta"}, "c2"),
        ("search_records", {"topic": "alpha"}, "c3"),
    ]
    tool_msgs, executor = await _run([_chunk(calls)])

    assert [c["args"]["topic"] for c in executor.calls] == ["alpha", "beta"]
    assert tool_msgs[0].content == tool_msgs[2].content  # c3 reused c1's result
    assert tool_msgs[1].content != tool_msgs[0].content


@pytest.mark.asyncio
async def test_cap_stops_execution_but_still_answers_every_call():
    calls = [("search_records", {"topic": f"t-{i}"}, f"c{i}") for i in range(12)]
    tool_msgs, executor = await _run([_chunk(calls)], max_tool_calls=3)

    assert len(executor.calls) == 3
    assert [c["args"]["topic"] for c in executor.calls] == ["t-0", "t-1", "t-2"]
    assert len(tool_msgs) == 12
    refused = [m for m in tool_msgs if _code(m) == "TOO_MANY_TOOL_CALLS"]
    assert len(refused) == 9


@pytest.mark.asyncio
async def test_confirmed_write_survives_the_cap():
    """The user clicked confirm; losing this call reads as "nothing was saved"."""
    calls = [("search_records", {"topic": f"t-{i}"}, f"c{i}") for i in range(10)]
    calls.append(("update_record", {"target": "r9", "proposal_token": "tok123"}, "write"))
    tool_msgs, executor = await _run([_chunk(calls)], max_tool_calls=3)

    dispatched = [c for c in executor.calls if c["name"] == "update_record"]
    assert len(dispatched) == 1
    # Real execute, not another proposal: the token rode along, dry_run did not.
    assert dispatched[0]["query"] is None
    assert dispatched[0]["args"]["proposal_token"] == "tok123"
    write_msg = next(m for m in tool_msgs if m.tool_call_id == "write")
    assert _envelope(write_msg)["ok"] is True


@pytest.mark.asyncio
async def test_unconfirmed_write_is_still_subject_to_the_cap():
    """Only the confirmed leg is protected — a proposal can be re-requested."""
    calls = [("search_records", {"topic": f"t-{i}"}, f"c{i}") for i in range(10)]
    calls.append(("update_record", {"target": "r9"}, "propose"))
    tool_msgs, executor = await _run([_chunk(calls)], max_tool_calls=3)

    assert not [c for c in executor.calls if c["name"] == "update_record"]
    refused = next(m for m in tool_msgs if m.tool_call_id == "propose")
    assert _code(refused) == "TOO_MANY_TOOL_CALLS"


# ------------------------------------------------------------- error envelopes


@pytest.mark.asyncio
async def test_bad_arguments_come_back_as_readable_json():
    """The model got the schema wrong. Its own error report has to survive the
    round trip — a pydantic ValidationError message is multi-line and quoted,
    which hand-rolled JSON string interpolation mangles into something the model
    can only guess at."""
    tool_msgs, executor = await _run([_chunk([("search_records", {"nonsense": 1}, "c1")])])

    assert executor.calls == []  # never dispatched; it failed validation first
    envelope = _envelope(tool_msgs[0])  # raises if the JSON is malformed
    assert envelope["error"]["code"] == "TOOL_EXCEPTION"
    assert "topic" in envelope["error"]["message"]  # names the missing field


@pytest.mark.asyncio
async def test_unknown_tool_comes_back_as_readable_json():
    tool_msgs, _ = await _run([_chunk([("no_such_tool", {}, "c1")])])

    assert _code(tool_msgs[0]) == "UNKNOWN_TOOL"
    assert "no_such_tool" in _envelope(tool_msgs[0])["error"]["message"]


@pytest.mark.asyncio
async def test_tool_exception_comes_back_as_readable_json():
    class _Boom:
        async def invoke(self, *, name, args, query=None, context=None):
            raise RuntimeError('backend said "no"\nand then some')

    tools = build_tools_from_catalog(CATALOG, executor=_Boom())
    graph = build_graph(
        tools,
        llm_factory=_ScriptedFactory(
            [_chunk([("search_records", {"topic": "x"}, "c1")]), AIMessageChunk(content="ok")]
        ),
        prompt_provider=_Prompt(),
    )
    state = await graph.ainvoke({"messages": [HumanMessage(content="hi")], "turns": 0})
    msg = next(m for m in state["messages"] if isinstance(m, ToolMessage))

    envelope = _envelope(msg)  # quotes and newlines must not break the envelope
    assert envelope["error"]["code"] == "TOOL_EXCEPTION"
    assert 'backend said "no"' in envelope["error"]["message"]


# ------------------------------------------------------------- truncation


@pytest.mark.asyncio
async def test_truncated_response_runs_none_of_its_tool_calls():
    """finish_reason=length means the array was cut off mid-generation, so it is
    incomplete by definition — executing it could fire a half-built call.

    Not reachable through api.openai.com, which errors the stream instead of
    delivering the partial array; an OpenAI-compatible gateway in front of the
    model is what makes this state real. Pinned so the guard survives the next
    refactor.
    """
    calls = [("search_records", {"topic": "alpha"}, "c1")]
    tool_msgs, executor = await _run([_chunk(calls, finish="length")])

    assert executor.calls == []
    assert tool_msgs == []


# ------------------------------------------------------------- end-to-end


def _token(sub: str = "u1") -> str:
    return jwt.encode(
        {"iss": "cogria-agent", "sub": sub, "exp": int(time.time()) + 300},
        SECRET,
        algorithm="HS256",
    )


def _chat_app(script: list[AIMessageChunk], backend, executor):
    config = AgentConfig(
        auth=AuthConfig(jwt_secret=SECRET, jwt_issuer="cogria-agent"),
        summarizer=SummarizerConfig(enabled=False),
    )
    return build_app(
        config,
        conversation_backend=backend,
        action_executor=executor,
        catalog_provider=StaticCatalogProvider(CATALOG),
        llm_factory=_ScriptedFactory([*script, AIMessageChunk(content="all done")]),
    )


def _post(client) -> tuple[str, str]:
    resp = client.post(
        "/chat", headers={"Authorization": f"Bearer {_token()}"}, json={"message": "hi"}
    )
    assert resp.status_code == 200
    cid = re.search(r'"conversation_id": "([0-9a-f-]{36})"', resp.text)
    assert cid, resp.text
    return cid.group(1), resp.text


def _rows(backend, conversation_id: str, *, expected: int, timeout: float = 2.0):
    """The turn is flushed by a background task; give it a moment to land."""
    import asyncio

    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = asyncio.run(backend.fetch_history(conversation_id, for_llm=False))
        if len(rows) >= expected:
            return rows
        time.sleep(0.05)
    raise AssertionError(f"only {len(rows)} rows persisted, wanted {expected}")


def test_results_pair_to_the_right_call_id_when_one_is_deduped():
    """Pairing used to be "first unresolved call with this name". Skip the
    middle call and that hands the third call's result to the second's id."""
    backend = InMemoryConversationBackend()
    executor = _RecordingExecutor()
    script = [
        _chunk(
            [
                ("search_records", {"topic": "alpha"}, "c1"),
                ("search_records", {"topic": "alpha"}, "c2"),  # dup -> skipped
                ("search_records", {"topic": "gamma"}, "c3"),
            ]
        )
    ]
    client = TestClient(_chat_app(script, backend, executor))
    cid, _ = _post(client)

    rows = _rows(backend, cid, expected=3)
    assistant = next(r for r in rows if r["role"] == "assistant")
    # All three are recorded as requested — the model did emit them.
    assert [c["id"] for c in assistant["content"]["tool_calls"]] == ["c1", "c2", "c3"]
    tool_rows = {
        r["content"]["tool_call_id"]: r["content"]["result"] for r in rows if r["role"] == "tool"
    }
    assert tool_rows["c1"]["data"]["n"] == 1
    assert tool_rows["c3"]["data"]["n"] == 2  # the second dispatch, not the first
    assert "c2" not in tool_rows  # never ran, nothing to record


def test_provider_finish_reason_is_persisted_and_truncation_announced():
    backend = InMemoryConversationBackend()
    executor = _RecordingExecutor()
    script = [_chunk([("search_records", {"topic": "alpha"}, "c1")], finish="length")]
    client = TestClient(_chat_app(script, backend, executor))
    cid, body = _post(client)

    # The client is told the reply was cut off, and no tool card is announced
    # for calls that will never run.
    assert "event: truncated" in body
    assert "event: tool_call" not in body

    rows = _rows(backend, cid, expected=2)
    assistant = next(r for r in rows if r["role"] == "assistant")
    # Previously this said "tool_calls" — indistinguishable from a healthy turn.
    assert assistant["finish_reason"] == "length"
    assert executor.calls == []


def test_ordinary_turn_still_reports_stop():
    backend = InMemoryConversationBackend()
    executor = _RecordingExecutor()
    script = [AIMessageChunk(content="just talking", response_metadata={"finish_reason": "stop"})]
    client = TestClient(_chat_app(script, backend, executor))
    cid, _ = _post(client)

    rows = _rows(backend, cid, expected=2)
    assistant = next(r for r in rows if r["role"] == "assistant")
    assert assistant["finish_reason"] == "stop"
