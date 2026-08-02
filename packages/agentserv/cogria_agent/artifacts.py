"""Artifact tools — front-end-defined tools the LLM calls to render something in
the right-hand panel. Unlike action tools, an artifact tool is a NO-OP on the
server: the `tool_call` SSE event itself is the render instruction, and the
runner returns an empty success envelope. The defs come from the front-end
registry via the BFF in the /chat body, so the front-end stays the single
source of truth.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from .tools import _build_args_model


def build_artifact_tools(defs: list[dict[str, Any]]) -> list[StructuredTool]:
    """Synthesize no-op StructuredTools from the front-end artifact registry."""
    tools: list[StructuredTool] = []
    for spec in defs or []:
        name = spec.get("name")
        if not name:
            continue
        args_model = _build_args_model(name, spec.get("params_schema") or {})

        async def _runner(**_kwargs: Any) -> str:
            # No execution: the SSE tool_call event already drove the panel.
            return '{"ok":true,"data":{"shown":true}}'

        tools.append(
            StructuredTool.from_function(
                coroutine=_runner,
                name=name,
                description=spec.get("description", ""),
                args_schema=args_model,
            )
        )
    return tools


def artifact_tool_names(defs: list[dict[str, Any]]) -> set[str]:
    """Names of the artifact tools in this request — used to route dispatch."""
    return {spec["name"] for spec in (defs or []) if spec.get("name")}
