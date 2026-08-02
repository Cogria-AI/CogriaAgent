"""Sample actions shared by the backend-py tests: a read + a write over an
in-memory store, using the AgentAction SDK."""

from __future__ import annotations

from typing import Any

from cogria_backend import AgentAction, AgentContext

STORE: dict[int, dict[str, Any]] = {}
_NEXT = {"id": 1}


def reset() -> None:
    STORE.clear()
    _NEXT["id"] = 1


class ListThings(AgentAction):
    name = "list_things"
    description = "List things."

    async def run(self, params: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
        return {"things": list(STORE.values())}

    def read_message(self, data, params):
        return f"{len(data['things'])} thing(s)."


class CreateThing(AgentAction):
    name = "create_thing"
    description = "Create a thing. Requires confirmation."
    requires_confirm = True
    target_type = "thing"

    def params_schema(self):
        return {
            "type": "object",
            "properties": {"label": {"type": "string", "description": "label"}},
            "required": ["label"],
            "additionalProperties": False,
        }

    async def handle(self, params, ctx):
        tid = _NEXT["id"]
        _NEXT["id"] += 1
        STORE[tid] = {"id": tid, "label": params["label"]}
        return {"id": tid, "label": params["label"]}

    def proposal_summary(self, params):
        return f'Create thing "{params["label"]}"'

    def success_message(self, result, params):
        return f'Created "{result["label"]}".'
