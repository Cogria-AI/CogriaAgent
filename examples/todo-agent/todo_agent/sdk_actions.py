"""The SAME todo capability, declared with the backend-py SDK (AgentAction
subclasses) instead of a hand-written executor. This is the "define your API"
path: write classes, register them — the SDK runs propose/confirm + audit, and
RegistryExecutor plugs into the agentserv kernel unchanged.
"""

from __future__ import annotations

from typing import Any

from cogria_backend import AgentAction, AgentContext, fail

from .actions import TodoStore

# Shared in-memory store for the SDK variant.
STORE = TodoStore()


class ListTodos(AgentAction):
    name = "list_todos"
    description = "List the user's todo items, optionally only the open ones. Use when the user asks what's on their list."

    def params_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"only_open": {"type": "boolean", "description": "If true, only not-yet-done items."}},
            "required": [],
            "additionalProperties": False,
        }

    async def run(self, params: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
        items = list(STORE.items.values())
        if params.get("only_open"):
            items = [t for t in items if not t.done]
        return {"todos": [{"id": t.id, "title": t.title, "done": t.done} for t in items]}

    def read_message(self, data, params):
        return f"{len(data['todos'])} todo(s)."


class AddTodo(AgentAction):
    name = "add_todo"
    description = "Add a new todo item. Requires confirmation. Use when the user wants to add a task."
    requires_confirm = True
    target_type = "todo"

    def params_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"title": {"type": "string", "description": "The task text."}},
            "required": ["title"],
            "additionalProperties": False,
        }

    async def handle(self, params, ctx):
        t = STORE.add(params["title"])
        return {"id": t.id, "title": t.title}

    def proposal_summary(self, params):
        return f'Add a todo: "{params["title"]}"'

    def success_message(self, result, params):
        return f'Added "{result["title"]}".'


class CompleteTodo(AgentAction):
    name = "complete_todo"
    description = "Mark a todo item done. Requires confirmation. Use when the user finished a task."
    requires_confirm = True
    target_type = "todo"

    def params_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"id": {"type": "integer", "description": "The todo id to complete."}},
            "required": ["id"],
            "additionalProperties": False,
        }

    async def validate_target(self, params, ctx):
        if int(params["id"]) not in STORE.items:
            return fail("NOT_FOUND", f"No todo with id {params['id']}.")
        return None

    async def handle(self, params, ctx):
        t = STORE.complete(int(params["id"]))
        return {"id": t.id, "done": True}

    def proposal_summary(self, params):
        return f"Mark todo #{params['id']} as done"

    def success_message(self, result, params):
        return f"Completed todo #{result['id']}."
