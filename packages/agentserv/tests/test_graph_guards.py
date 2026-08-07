"""Graph-level guards around the system prompt and the propose/confirm gate.

Two regressions pinned here:

1. The operating prompt must lead every LLM request. The old guard was
   `if not isinstance(msgs[0], SystemMessage)` — but the backends inject the
   running summary as a leading role=system row, so from the first fold onward
   the real prompt was never injected again and the agent ran on a 200-word
   conversation summary as its entire rulebook.

2. route_after_tools must scan the whole tail of tool results, not just the
   last one. A round calling [write_tool, read_tool] puts the proposal in the
   middle; stopping at the last ToolMessage let the graph run on, and the model
   narrated (or retried) a proposal the user had not confirmed yet.
"""

from __future__ import annotations

import json

import pytest
from cogria_agent.graph import build_graph
from cogria_agent.tools import build_tools_from_catalog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGenerationChunk

CATALOG = {
    "actions": [
        {
            "name": "search_records",
            "description": "Search records.",
            "params_schema": {
                "type": "object",
                "properties": {"topic": {"type": "string", "description": "what to find"}},
                "required": ["topic"],
            },
            "requires_confirm": False,
        },
        {
            "name": "update_record",
            "description": "Update a record. Write action.",
            "params_schema": {
                "type": "object",
                "properties": {"target": {"type": "string", "description": "record id"}},
                "required": ["target"],
            },
            "requires_confirm": True,
        },
    ]
}


class _Executor:
    """Returns a proposal for dry_run write calls, a plain envelope otherwise."""

    async def invoke(self, *, name, args, query=None, context=None):
        if query and query.get("dry_run"):
            return {
                "ok": True,
                "data": {"requires_confirm": True, "proposal_token": "tok-1", "summary": "s"},
            }
        return {"ok": True, "data": {}}


class _RecordingLLM(BaseChatModel):
    """Replays one scripted chunk per round and records every request's messages."""

    script: list
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "recording"

    def bind_tools(self, tools, **kwargs):
        return self

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        chunk = self.script.pop(0) if self.script else AIMessageChunk(content="done")
        yield ChatGenerationChunk(message=chunk)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("streaming only")


class _Factory:
    def __init__(self, script: list[AIMessageChunk]) -> None:
        self.llm = _RecordingLLM(script=script, seen=[])

    def chat_llm(self, model=None):
        return self.llm

    def summary_llm(self):
        return self.llm

    def model_name(self):
        return "test-model"


class _Prompt:
    def system_prompt(self, *, locale=None) -> str:
        return "OPERATING PROMPT"


def _calls_chunk(calls: list[tuple[str, dict, str]]) -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": name,
                "args": json.dumps(args),
                "id": cid,
                "index": i,
                "type": "tool_call_chunk",
            }
            for i, (name, args, cid) in enumerate(calls)
        ],
        response_metadata={"finish_reason": "tool_calls"},
    )


def _graph(factory: _Factory):
    tools = build_tools_from_catalog(CATALOG, executor=_Executor())
    return build_graph(tools, llm_factory=factory, prompt_provider=_Prompt())


# ------------------------------------------------------- system prompt leads


@pytest.mark.asyncio
async def test_operating_prompt_leads_even_when_a_summary_row_is_first():
    """A folded conversation replays as [system(summary), ...]. The old
    isinstance check saw a SystemMessage in front and skipped injection — the
    summary WAS the rulebook from then on."""
    factory = _Factory([AIMessageChunk(content="hello")])
    graph = _graph(factory)
    await graph.ainvoke(
        {
            "messages": [
                SystemMessage(content="summary of the earlier conversation"),
                HumanMessage(content="hi"),
            ],
            "turns": 0,
        }
    )

    request = factory.llm.seen[0]
    assert request[0].content == "OPERATING PROMPT"
    # The summary is still there — demoted to context, not dropped.
    assert request[1].content == "summary of the earlier conversation"


@pytest.mark.asyncio
async def test_operating_prompt_leads_on_every_round_of_a_tool_turn():
    factory = _Factory(
        [
            _calls_chunk([("search_records", {"topic": "alpha"}, "c1")]),
            AIMessageChunk(content="answer"),
        ]
    )
    graph = _graph(factory)
    await graph.ainvoke({"messages": [HumanMessage(content="hi")], "turns": 0})

    assert len(factory.llm.seen) == 2
    for request in factory.llm.seen:
        assert request[0].content == "OPERATING PROMPT"
        # Injected once per request, never stacked.
        assert sum(1 for m in request if m.content == "OPERATING PROMPT") == 1


# --------------------------------------------------- propose/confirm routing


@pytest.mark.asyncio
async def test_proposal_mid_round_still_ends_the_turn():
    """[write, read] in one round: the proposal lands mid-tail. The turn must
    END for the ConfirmCard — one LLM round, no narration of the proposal."""
    factory = _Factory(
        [
            _calls_chunk(
                [
                    ("update_record", {"target": "r1"}, "w1"),  # dry_run -> proposal
                    ("search_records", {"topic": "alpha"}, "r2"),
                ]
            )
        ]
    )
    graph = _graph(factory)
    state = await graph.ainvoke({"messages": [HumanMessage(content="hi")], "turns": 0})

    assert len(factory.llm.seen) == 1  # never went back to the model
    assert isinstance(state["messages"][-1], ToolMessage)


@pytest.mark.asyncio
async def test_proposal_last_in_round_ends_the_turn():
    factory = _Factory([_calls_chunk([("update_record", {"target": "r1"}, "w1")])])
    graph = _graph(factory)
    state = await graph.ainvoke({"messages": [HumanMessage(content="hi")], "turns": 0})

    assert len(factory.llm.seen) == 1
    assert isinstance(state["messages"][-1], ToolMessage)


@pytest.mark.asyncio
async def test_round_without_proposal_returns_to_the_model():
    factory = _Factory(
        [
            _calls_chunk([("search_records", {"topic": "alpha"}, "c1")]),
            AIMessageChunk(content="answer"),
        ]
    )
    graph = _graph(factory)
    state = await graph.ainvoke({"messages": [HumanMessage(content="hi")], "turns": 0})

    assert len(factory.llm.seen) == 2
    assert isinstance(state["messages"][-1], AIMessage)
