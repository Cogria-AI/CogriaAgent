"""Dynamic tool synthesis + propose/confirm (dry_run) routing."""

from __future__ import annotations

import json

import pytest

from cogria_agent.tools import build_tools_from_catalog

CATALOG = {
    "actions": [
        {
            "name": "list_things",
            "description": "List things.",
            "params_schema": {"type": "object", "properties": {}, "required": []},
            "requires_confirm": False,
        },
        {
            "name": "delete_thing",
            "description": "Delete a thing. Requires confirmation.",
            "params_schema": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "id"}},
                "required": ["id"],
            },
            "requires_confirm": True,
        },
    ]
}


class RecordingExecutor:
    """Implements protocols.ActionExecutor; records every invoke + returns canned."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def invoke(self, *, name, args, query=None, context=None):
        self.calls.append({"name": name, "args": args, "query": query})
        if query and query.get("dry_run"):
            return {"ok": True, "data": {"requires_confirm": True, "proposal_token": "tok123", "summary": "will delete"}}
        return {"ok": True, "data": {"id": args.get("id")}, "message": "done"}


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


@pytest.mark.asyncio
async def test_read_tool_invokes_without_dry_run():
    ex = RecordingExecutor()
    tools = build_tools_from_catalog(CATALOG, executor=ex)
    out = await _tool(tools, "list_things").ainvoke({})
    assert json.loads(out)["ok"] is True
    assert ex.calls == [{"name": "list_things", "args": {}, "query": None}]


@pytest.mark.asyncio
async def test_write_tool_first_call_is_dry_run_propose():
    ex = RecordingExecutor()
    tools = build_tools_from_catalog(CATALOG, executor=ex)
    out = await _tool(tools, "delete_thing").ainvoke({"id": 42})
    env = json.loads(out)
    assert env["data"]["proposal_token"] == "tok123"
    # first call: dry_run set, no proposal_token leaked into args
    assert ex.calls[0]["query"] == {"dry_run": "1"}
    assert "proposal_token" not in ex.calls[0]["args"]
    assert ex.calls[0]["args"]["id"] == 42


@pytest.mark.asyncio
async def test_write_tool_with_token_executes():
    ex = RecordingExecutor()
    tools = build_tools_from_catalog(CATALOG, executor=ex)
    out = await _tool(tools, "delete_thing").ainvoke({"id": 42, "proposal_token": "tok123"})
    env = json.loads(out)
    assert env["message"] == "done"
    assert ex.calls[0]["query"] is None
    assert ex.calls[0]["args"]["proposal_token"] == "tok123"


def test_write_tool_schema_has_proposal_token_field():
    ex = RecordingExecutor()
    tools = build_tools_from_catalog(CATALOG, executor=ex)
    fields = _tool(tools, "delete_thing").args_schema.model_fields
    assert "proposal_token" in fields
    assert "proposal_token" not in _tool(tools, "list_things").args_schema.model_fields


@pytest.mark.asyncio
async def test_stringified_object_and_array_args_are_parsed():
    """qwen (and other OpenAI-compatible models) may serialize nested object/
    array arguments as JSON strings; validation must undo that, not reject."""
    catalog = {
        "actions": [
            {
                "name": "log_record",
                "description": "Log.",
                "params_schema": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "payload": {"type": "object"},
                        "tags": {"type": "array"},
                    },
                    "required": ["type"],
                },
                "requires_confirm": False,
            }
        ]
    }
    ex = RecordingExecutor()
    tools = build_tools_from_catalog(catalog, executor=ex)
    await _tool(tools, "log_record").ainvoke(
        {"type": "coffee", "payload": '{"count": 1}', "tags": '["a", "b"]'}
    )
    assert ex.calls[0]["args"]["payload"] == {"count": 1}
    assert ex.calls[0]["args"]["tags"] == ["a", "b"]

    # real objects still pass through untouched; non-JSON strings still fail
    await _tool(tools, "log_record").ainvoke({"type": "water", "payload": {"count": 2}})
    assert ex.calls[1]["args"]["payload"] == {"count": 2}
    with pytest.raises(Exception):
        await _tool(tools, "log_record").ainvoke({"type": "water", "payload": "not json"})
