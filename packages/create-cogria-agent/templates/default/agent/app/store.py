"""Demo data store. Replace with your real database/models — the agent kernel
never touches this; only your actions (actions.py) do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PRIORITIES = ["low", "medium", "high"]


@dataclass
class Task:
    id: int
    title: str
    priority: str = "medium"
    done: bool = False


@dataclass
class TaskStore:
    items: dict[int, Task] = field(default_factory=dict)
    _next: int = 1

    def add(self, title: str, priority: str = "medium") -> Task:
        task = Task(id=self._next, title=title, priority=priority)
        self.items[task.id] = task
        self._next += 1
        return task

    def get(self, task_id: int) -> Task | None:
        return self.items.get(task_id)

    def list(self) -> list[Task]:
        return list(self.items.values())
