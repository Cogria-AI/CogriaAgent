"""Registry of agent actions. Holds one instance per name and produces the
catalog. Register explicitly, or `discover()` a module."""

from __future__ import annotations

import inspect
from typing import Any

from .action import AgentAction


class Registry:
    def __init__(self, actions: list[AgentAction] | None = None) -> None:
        self._by_name: dict[str, AgentAction] = {}
        for a in actions or []:
            self.register(a)

    def register(self, action: AgentAction) -> Registry:
        if not getattr(action, "name", ""):
            raise ValueError(f"{type(action).__name__} is missing a non-empty `name`.")
        if action.name in self._by_name:
            raise ValueError(f"Duplicate action name '{action.name}'.")
        self._by_name[action.name] = action
        return self

    def discover(self, module: Any) -> Registry:
        """Register every concrete AgentAction subclass defined in a module."""
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, AgentAction) and obj is not AgentAction and not inspect.isabstract(obj):
                if obj.__module__ == module.__name__ and getattr(obj, "name", ""):
                    self.register(obj())
        return self

    def all(self) -> list[AgentAction]:
        return list(self._by_name.values())

    def find_by_name(self, name: str) -> AgentAction | None:
        return self._by_name.get(name)

    def find_by_slug(self, slug: str) -> AgentAction | None:
        return next((a for a in self._by_name.values() if a.url_slug() == slug), None)

    def catalog(self) -> dict[str, Any]:
        return {"actions": [a.catalog_entry() for a in self._by_name.values()]}
