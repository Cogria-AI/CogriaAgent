"""Your business actions — the one place you write code to teach the agent what
it can do.

Each action declares: a snake_case `name` (the tool name the LLM sees), a natural
language `description` (how the LLM decides to use it), a `params_schema` (JSON
Schema, used to synthesize + validate the tool), and either `run()` (reads) or
`handle()` + `requires_confirm = True` (writes, routed through propose/confirm).

Replace these three task actions with your own domain. The kernel doesn't change.
"""

from __future__ import annotations

from typing import Any

from cogria_backend import AgentAction, AgentContext, fail

from .store import PRIORITIES, TaskStore

# Demo store, shared for the process. Swap for your real services.
STORE = TaskStore()


def _view(t) -> dict[str, Any]:
    return {"id": t.id, "title": t.title, "priority": t.priority, "done": t.done}


class ListTasks(AgentAction):
    name = "list_tasks"
    description = (
        "List the user's tasks. Use when the user asks what's on their list or to "
        "review their tasks."
    )

    def params_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "only_open": {"type": "boolean", "description": "If true, only not-yet-done tasks."}
            },
            "required": [],
            "additionalProperties": False,
        }

    async def run(self, params: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
        items = STORE.list()
        if params.get("only_open"):
            items = [t for t in items if not t.done]
        return {"tasks": [_view(t) for t in items]}

    def read_message(self, data, params):
        return f"{len(data['tasks'])} task(s)."


class AddTask(AgentAction):
    name = "add_task"
    description = (
        "Add a new task. Requires confirmation. Use when the user wants to add "
        "something to their list."
    )
    requires_confirm = True
    target_type = "task"

    def params_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The task text."},
                "priority": {"type": "string", "enum": PRIORITIES, "description": "Optional priority."},
            },
            "required": ["title"],
            "additionalProperties": False,
        }

    async def handle(self, params, ctx):
        t = STORE.add(params["title"], params.get("priority", "medium"))
        return _view(t)

    def proposal_summary(self, params):
        pr = params.get("priority", "medium")
        return f'Add task "{params["title"]}" ({pr} priority)'

    def success_message(self, result, params):
        return f'Added "{result["title"]}".'


class SetPriority(AgentAction):
    name = "set_priority"
    description = (
        "Change a task's priority (low/medium/high). Requires confirmation. Use when "
        "the user wants to reprioritize a task."
    )
    requires_confirm = True
    target_type = "task"

    def params_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "The task id."},
                "priority": {"type": "string", "enum": PRIORITIES, "description": "New priority."},
            },
            "required": ["id", "priority"],
            "additionalProperties": False,
        }

    async def validate_target(self, params, ctx):
        if STORE.get(int(params["id"])) is None:
            return fail("NOT_FOUND", f"No task with id {params['id']}.")
        return None

    async def capture_before(self, params, ctx):
        t = STORE.get(int(params["id"]))
        return {"priority": t.priority} if t else None

    async def handle(self, params, ctx):
        t = STORE.get(int(params["id"]))
        t.priority = params["priority"]
        return _view(t)

    async def capture_after(self, params, ctx, result):
        return {"priority": result["priority"]}

    def summarize(self, before, after, params):
        if before and after:
            return f"priority {before['priority']} → {after['priority']}"
        return f"set priority of task #{params.get('id')}"

    def proposal_summary(self, params):
        return f"Set task #{params['id']} to {params['priority']} priority"

    def success_message(self, result, params):
        return f'Task "{result["title"]}" is now {result["priority"]} priority.'


def all_actions() -> list[AgentAction]:
    return [ListTasks(), AddTask(), SetPriority()]
