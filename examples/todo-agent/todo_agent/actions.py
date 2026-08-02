"""A tiny business domain: an in-memory todo store + an ActionExecutor.

This is what an embedding project provides when it runs actions in-process. Put
it next to examples/menu-agent — a completely different domain on the same
kernel — and the point lands: the kernel carries no domain assumptions.

Implements protocols.ActionExecutor. Propose/confirm is handled inline here; the
backend-py SDK provides the same orchestration from a base class.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Todo:
    id: int
    title: str
    done: bool = False


@dataclass
class TodoStore:
    items: dict[int, _Todo] = field(default_factory=dict)
    _next: int = 1

    def add(self, title: str) -> _Todo:
        todo = _Todo(id=self._next, title=title)
        self.items[todo.id] = todo
        self._next += 1
        return todo

    def complete(self, todo_id: int) -> _Todo | None:
        t = self.items.get(todo_id)
        if t:
            t.done = True
        return t


# Catalog the kernel advertises to the LLM. snake_case names; writes set
# requires_confirm so the kernel routes them through propose/confirm.
CATALOG: dict[str, Any] = {
    "actions": [
        {
            "name": "list_todos",
            "description": "List the user's todo items, optionally filtered. Use when the user asks what's on their list.",
            "params_schema": {
                "type": "object",
                "properties": {
                    "only_open": {"type": "boolean", "description": "If true, only return not-yet-done items."}
                },
                "required": [],
            },
            "requires_confirm": False,
        },
        {
            "name": "add_todo",
            "description": "Add a new todo item. Requires confirmation. Use when the user wants to add a task.",
            "params_schema": {
                "type": "object",
                "properties": {"title": {"type": "string", "description": "The task text."}},
                "required": ["title"],
            },
            "requires_confirm": True,
        },
        {
            "name": "complete_todo",
            "description": "Mark a todo item done. Requires confirmation. Use when the user finished a task.",
            "params_schema": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "The todo id to complete."}},
                "required": ["id"],
            },
            "requires_confirm": True,
        },
    ]
}


class TodoExecutor:
    """Implements protocols.ActionExecutor with inline propose/confirm."""

    def __init__(self, store: TodoStore | None = None) -> None:
        self.store = store or TodoStore()
        self._pending: dict[str, tuple[str, dict[str, Any]]] = {}  # token -> (name, args)

    async def invoke(
        self,
        *,
        name: str,
        args: dict[str, Any],
        query: dict[str, str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if name == "list_todos":
            return self._list(args)

        # write actions: propose (dry_run) vs confirm (token present)
        if query and query.get("dry_run"):
            token = uuid.uuid4().hex[:16]
            self._pending[token] = (name, {k: v for k, v in args.items() if k != "proposal_token"})
            return {
                "ok": True,
                "data": {"requires_confirm": True, "proposal_token": token, "summary": self._summary(name, args)},
                "message": self._summary(name, args),
            }

        token = args.pop("proposal_token", None)
        if not token:
            return {"ok": False, "error": {"code": "PROPOSAL_REQUIRED", "message": "Confirmation required."}}
        pending = self._pending.pop(token, None)
        if not pending or pending[0] != name:
            return {"ok": False, "error": {"code": "PROPOSAL_EXPIRED", "message": "Proposal expired or invalid."}}

        return self._execute(name, pending[1])

    # --- helpers ---------------------------------------------------------
    def _list(self, args: dict[str, Any]) -> dict[str, Any]:
        items = self.store.items.values()
        if args.get("only_open"):
            items = [t for t in items if not t.done]
        data = [{"id": t.id, "title": t.title, "done": t.done} for t in items]
        return {"ok": True, "data": {"todos": data}, "message": f"{len(data)} todo(s)."}

    def _execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "add_todo":
            t = self.store.add(args["title"])
            return {"ok": True, "data": {"id": t.id, "title": t.title}, "message": f"Added '{t.title}'."}
        if name == "complete_todo":
            t = self.store.complete(int(args["id"]))
            if not t:
                return {"ok": False, "error": {"code": "NOT_FOUND", "message": "No such todo."}}
            return {"ok": True, "data": {"id": t.id, "done": True}, "message": f"Completed '{t.title}'."}
        return {"ok": False, "error": {"code": "UNKNOWN_ACTION", "message": name}}

    def _summary(self, name: str, args: dict[str, Any]) -> str:
        if name == "add_todo":
            return f"Add a todo: \"{args.get('title', '')}\""
        if name == "complete_todo":
            return f"Mark todo #{args.get('id')} as done"
        return name
