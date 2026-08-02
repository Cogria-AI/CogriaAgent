"""Dynamically build LangChain StructuredTools from a catalog.

Each catalog action becomes one LangChain tool. The LLM only sees
name/description/params_schema — it never supplies auth or routing; those are
the ActionExecutor's concern. Propose/confirm (dry_run + proposal_token) is
handled here so every action gets it uniformly.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from .protocols import ActionExecutor

# JSON-Schema type -> python type. Minimal mapping; expand as catalogs grow.
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _build_args_model(
    action_name: str, schema: dict[str, Any], *, requires_confirm: bool = False
) -> type[BaseModel]:
    """Synthesize a pydantic model for an action's params_schema.

    Empty schema -> empty model (e.g. ping). For write actions we add an optional
    `proposal_token` so the LLM can re-supply it on the [CONFIRMED] turn.
    """
    properties: dict[str, Any] = schema.get("properties") or {}
    required: set[str] = set(schema.get("required") or [])

    fields: dict[str, tuple[type, Any]] = {}
    for prop_name, prop_schema in properties.items():
        py_type = _TYPE_MAP.get(prop_schema.get("type", "string"), str)
        description = prop_schema.get("description", "")
        if prop_name in required:
            fields[prop_name] = (py_type, Field(..., description=description))
        else:
            fields[prop_name] = (py_type | None, Field(default=None, description=description))

    if requires_confirm:
        fields["proposal_token"] = (
            str | None,
            Field(
                default=None,
                description=(
                    "Leave empty on the first call (it returns a proposal to confirm). "
                    "Only set this to the proposal_token from a prior proposal after the "
                    "user has confirmed (i.e. when handling a [CONFIRMED] message)."
                ),
            ),
        )

    model_name = f"{action_name.title().replace('_', '')}Args"
    if not fields:
        return create_model(model_name, __config__=ConfigDict(extra="forbid"))
    return create_model(model_name, __config__=ConfigDict(extra="forbid"), **fields)  # type: ignore[arg-type]


def build_tools_from_catalog(
    catalog: dict[str, Any], *, executor: ActionExecutor, context: dict[str, Any] | None = None
) -> list[StructuredTool]:
    """One StructuredTool per catalog action, dispatching through the executor.
    `context` (per-request identity) is forwarded to every invoke."""
    return [_build_one_tool(spec, executor=executor, context=context) for spec in catalog.get("actions", [])]


def _build_one_tool(spec: dict[str, Any], *, executor: ActionExecutor, context: dict[str, Any] | None) -> StructuredTool:
    name: str = spec["name"]
    description: str = spec.get("description", "")
    requires_confirm: bool = bool(spec.get("requires_confirm", False))
    args_model = _build_args_model(name, spec.get("params_schema") or {}, requires_confirm=requires_confirm)

    async def _runner(**kwargs: Any) -> str:
        # Propose/confirm (MVP-A): first call to a write action has no token ->
        # send dry_run=1 so the backend returns a proposal_token + summary instead
        # of mutating. The [CONFIRMED] turn supplies the token -> real execute.
        # Read actions ignore all of this.
        query: dict[str, str] | None = None
        token = kwargs.pop("proposal_token", None) if requires_confirm else None
        if requires_confirm and not token:
            query = {"dry_run": "1"}
        elif requires_confirm and token:
            kwargs["proposal_token"] = token

        envelope = await executor.invoke(name=name, args=kwargs, query=query, context=context)
        # LangChain tools must return a string; hand back the JSON envelope so the
        # model can read message / error code / proposal_token.
        return json.dumps(envelope, ensure_ascii=False)

    return StructuredTool.from_function(
        coroutine=_runner,
        name=name,
        description=description,
        args_schema=args_model,
    )
