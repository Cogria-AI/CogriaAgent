"""The restaurant business, declared with the backend-py SDK.

A realistic write-heavy domain — "create dish / change price / toggle
availability / list menu" — expressed as `AgentAction` subclasses. The SDK runs
propose/confirm, audit, and envelope translation; `RegistryExecutor` plugs the
registry into the
agentserv kernel **without touching the kernel** — that's the whole point of the
example: swap the domain (todo → restaurant), kernel unchanged.
"""

from __future__ import annotations

from typing import Any

from cogria_backend import AgentAction, AgentContext, fail

from .store import MenuStore

# One shared menu for the process. A real action would reach into your services.
STORE = MenuStore().seed()

_CATEGORIES = ["Starters", "Mains", "Desserts", "Drinks"]


def _dish_view(d) -> dict[str, Any]:
    return {"id": d.id, "name": d.name, "price": d.price, "category": d.category, "available": d.available}


class ListDishes(AgentAction):
    name = "list_dishes"
    description = (
        "List dishes on the menu, optionally filtered by category or to available "
        "items only. Use when the user asks what's on the menu or to review dishes."
    )

    def params_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": _CATEGORIES,
                    "description": "Only dishes in this menu category.",
                },
                "available_only": {
                    "type": "boolean",
                    "description": "If true, only dishes currently available to order.",
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    async def run(self, params: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
        dishes = STORE.list(params.get("category"), bool(params.get("available_only")))
        return {"dishes": [_dish_view(d) for d in dishes]}

    def read_message(self, data, params):
        return f"{len(data['dishes'])} dish(es) on the menu."


class CreateDish(AgentAction):
    name = "create_dish"
    description = (
        "Add a new dish to the menu. Requires confirmation. Use when the user wants "
        "to create or add a menu item with a name, price and category."
    )
    requires_confirm = True
    target_type = "dish"
    ability = "dishes.create"

    def params_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Dish name, e.g. 'Margherita Pizza'."},
                "price": {"type": "number", "minimum": 0, "description": "Price in the menu currency."},
                "category": {"type": "string", "enum": _CATEGORIES, "description": "Menu category."},
            },
            "required": ["name", "price", "category"],
            "additionalProperties": False,
        }

    async def handle(self, params, ctx):
        d = STORE.add(params["name"], float(params["price"]), params["category"])
        return _dish_view(d)

    def proposal_summary(self, params):
        return f'Add "{params["name"]}" to {params["category"]} at {params["price"]:.2f}'

    def success_message(self, result, params):
        return f'Added "{result["name"]}" ({result["category"]}, {result["price"]:.2f}).'


class UpdatePrice(AgentAction):
    name = "update_price"
    description = (
        "Change the price of an existing dish. Requires confirmation. Use when the "
        "user wants to re-price a menu item."
    )
    requires_confirm = True
    target_type = "dish"
    ability = "dishes.update"

    def params_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "The dish id to re-price."},
                "price": {"type": "number", "minimum": 0, "description": "The new price."},
            },
            "required": ["id", "price"],
            "additionalProperties": False,
        }

    async def validate_target(self, params, ctx):
        if STORE.get(int(params["id"])) is None:
            return fail("NOT_FOUND", f"No dish with id {params['id']}.")
        return None

    async def capture_before(self, params, ctx):
        d = STORE.get(int(params["id"]))
        return {"price": d.price} if d else None

    async def handle(self, params, ctx):
        d = STORE.get(int(params["id"]))
        d.price = float(params["price"])
        return _dish_view(d)

    async def capture_after(self, params, ctx, result):
        return {"price": result["price"]}

    def summarize(self, before, after, params):
        if before and after:
            return f"price {before['price']:.2f} → {after['price']:.2f}"
        return f"set price of dish #{params.get('id')}"

    def proposal_summary(self, params):
        return f"Re-price dish #{params['id']} to {params['price']:.2f}"

    def success_message(self, result, params):
        return f'"{result["name"]}" is now {result["price"]:.2f}.'


class SetAvailability(AgentAction):
    name = "set_availability"
    description = (
        "Mark a dish as available or unavailable (sold out). Requires confirmation. "
        "Use when the user wants to take a dish off or put it back on the menu."
    )
    requires_confirm = True
    target_type = "dish"
    ability = "dishes.update"

    def params_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "The dish id."},
                "available": {"type": "boolean", "description": "True = orderable, False = sold out."},
            },
            "required": ["id", "available"],
            "additionalProperties": False,
        }

    async def validate_target(self, params, ctx):
        if STORE.get(int(params["id"])) is None:
            return fail("NOT_FOUND", f"No dish with id {params['id']}.")
        return None

    async def handle(self, params, ctx):
        d = STORE.get(int(params["id"]))
        d.available = bool(params["available"])
        return _dish_view(d)

    def proposal_summary(self, params):
        state = "available" if params["available"] else "sold out"
        return f"Mark dish #{params['id']} as {state}"

    def success_message(self, result, params):
        state = "available" if result["available"] else "sold out"
        return f'"{result["name"]}" is now {state}.'


def all_actions() -> list[AgentAction]:
    return [ListDishes(), CreateDish(), UpdatePrice(), SetAvailability()]
