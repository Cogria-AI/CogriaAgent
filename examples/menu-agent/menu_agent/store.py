"""In-memory restaurant menu — the "business database" this example stands in for.

In a real deployment this would be your Laravel/Django/Rails models. Here it's a
dict so the example runs with zero infrastructure. The agent kernel never sees
this; it only ever talks to the actions in actions.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Dish:
    id: int
    name: str
    price: float
    category: str
    available: bool = True


@dataclass
class MenuStore:
    items: dict[int, Dish] = field(default_factory=dict)
    _next: int = 1

    def seed(self) -> MenuStore:
        """A few starter dishes so a fresh demo isn't an empty menu."""
        self.add("Margherita Pizza", 12.5, "Mains")
        self.add("Caesar Salad", 8.0, "Starters")
        self.add("Tiramisu", 6.5, "Desserts")
        return self

    def add(self, name: str, price: float, category: str) -> Dish:
        dish = Dish(id=self._next, name=name, price=price, category=category)
        self.items[dish.id] = dish
        self._next += 1
        return dish

    def get(self, dish_id: int) -> Dish | None:
        return self.items.get(dish_id)

    def list(self, category: str | None = None, available_only: bool = False) -> list[Dish]:
        dishes = list(self.items.values())
        if category:
            dishes = [d for d in dishes if d.category.lower() == category.lower()]
        if available_only:
            dishes = [d for d in dishes if d.available]
        return dishes
