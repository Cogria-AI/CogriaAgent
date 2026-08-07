"""Minimal LangGraph chat graph: LLM -> optional tool -> LLM -> END.

The model can call any tool (action or artifact) in one or more rounds; we cap
iterations to max_turns so a malformed tool loop can't burn budget. The system
prompt + reply language come from a SystemPromptProvider, and the LLM from an
LLMFactory — no business prompt or locale set is hardcoded here.

max_turns bounds the number of rounds, not the width of any one of them: a
single degenerate response can carry dozens of calls, and those all execute
before the round counter is next consulted. Actions can cost the user real
money (a paid external API, credits deducted per call), so tool_node also
dedups and caps within the response — see _select_calls.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .protocols import SystemPromptProvider


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    turns: int  # incremented before each LLM call; guards against > max_turns


def build_graph(
    tools: list[BaseTool],
    *,
    llm_factory: Any,
    prompt_provider: SystemPromptProvider,
    locale: str | None = None,
    max_turns: int = 10,
    max_tool_calls: int = 8,
    prompt_suffix: str = "",
    model_override: str | None = None,
):
    """`prompt_suffix` appends per-request guidance (e.g. how to treat attachment
    content) without projects having to bake it into their own system prompt.
    `model_override` swaps the chat model for this request only — used to route
    a turn carrying images to a vision-capable model. `max_tool_calls` caps how
    many distinct calls one model response may actually execute."""
    llm = (
        llm_factory.chat_llm(model=model_override) if model_override else llm_factory.chat_llm()
    ).bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}
    system_prompt = prompt_provider.system_prompt(locale=locale) + prompt_suffix

    async def chat_node(state: ChatState) -> dict[str, Any]:
        msgs = _with_system_prompt(state["messages"], system_prompt)

        accumulator: AIMessageChunk | None = None
        async for chunk in llm.astream(msgs):
            accumulator = chunk if accumulator is None else accumulator + chunk

        turns = state.get("turns", 0) + 1
        if accumulator is None:
            return {"messages": [AIMessage(content="")], "turns": turns}

        meta = accumulator.response_metadata or {}
        calls = getattr(accumulator, "tool_calls", []) or []
        if calls and meta.get("finish_reason") == "length":
            # Generation was cut off partway through the tool_calls array, so by
            # definition it is incomplete — the last call's arguments are
            # half-written and the ones before it are whatever happened to be
            # emitted, not a plan the model finished thinking through. Drop the
            # calls; the text still stands.
            #
            # Deliberately unreachable against api.openai.com, which raises
            # ("Could not finish the tool call because max_tokens was reached")
            # instead of handing back the partial array. An OpenAI-compatible
            # gateway in front of the model is what makes this state real: one
            # deployment received a truncated array as if it were normal output
            # and executed all 46 calls in it.
            calls = []
        final = AIMessage(
            content=accumulator.content,
            tool_calls=calls,
            # Carried through so the caller can persist the provider's real
            # finish_reason instead of inferring one from local state.
            response_metadata=meta,
        )
        return {"messages": [final], "turns": turns}

    async def tool_node(state: ChatState) -> dict[str, Any]:
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", []) or [])
        allowed = _select_calls(calls, max_tool_calls)

        results: list[ToolMessage] = []
        executed: dict[str, str] = {}  # call key -> content of its one execution
        for call in calls:
            key = _call_key(call)
            if key in executed:
                # Same tool, same args, same response: hand back the first
                # result verbatim. The model sees exactly what it would have
                # seen, while the action (and its charge) happens once.
                results.append(ToolMessage(content=executed[key], tool_call_id=call["id"]))
                continue
            if key not in allowed:
                results.append(ToolMessage(content=_TOO_MANY_CALLS, tool_call_id=call["id"]))
                continue

            tool = tools_by_name.get(call["name"])
            if not tool:
                content = _error_envelope("UNKNOWN_TOOL", f"No such tool: {call['name']}")
            else:
                try:
                    # Invoked with the whole tool_call (not just its args) so the
                    # returned ToolMessage carries tool_call_id — that is what
                    # lets the caller pair results to calls by id rather than by
                    # arrival order, which skipped calls would throw off.
                    payload = await tool.ainvoke({**call, "type": "tool_call"})
                    content = payload.content if isinstance(payload, ToolMessage) else str(payload)
                except Exception as e:  # noqa: BLE001 — tool failures must not crash the graph
                    content = _error_envelope("TOOL_EXCEPTION", f"{type(e).__name__}: {e}")
            executed[key] = content
            results.append(ToolMessage(content=content, tool_call_id=call["id"]))
        return {"messages": results}

    def route_after_chat(state: ChatState) -> str:
        if state.get("turns", 0) >= max_turns:
            return END
        last = state["messages"][-1] if state["messages"] else None
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    def route_after_tools(state: ChatState) -> str:
        # Propose/confirm gate (MVP-A): if a write action just returned a proposal
        # (dry_run), END the turn so the front-end can show the ConfirmCard rather
        # than letting the LLM narrate the proposal before the user confirms.
        #
        # Scan every result from THIS round, not just the last one: a round
        # calling [write_tool, read_tool] puts the proposal in the middle, and
        # stopping at the last ToolMessage let the graph run on — the model then
        # saw a raw requires_confirm envelope it was told not to narrate and
        # retried the tool. A round's results are contiguous at the tail, so
        # stop at the first non-tool message.
        for msg in reversed(state["messages"]):
            if not isinstance(msg, ToolMessage):
                break
            if _is_proposal(msg.content):
                return END
        return "chat"

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("chat")
    graph.add_conditional_edges("chat", route_after_chat, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_after_tools, {"chat": "chat", END: END})
    return graph.compile()


def _with_system_prompt(messages: list[BaseMessage], system_prompt: str) -> list[BaseMessage]:
    """Guarantee the operating prompt leads every request to the LLM.

    This used to be `if not isinstance(messages[0], SystemMessage)`, which
    silently swapped the agent's entire rulebook for a conversation summary:
    the backends inject the running summary as a leading role=system row, so
    from the first fold onward messages[0] was always a SystemMessage and the
    real prompt was never injected again — the agent lost the propose/confirm
    discipline mid-conversation. Replayed system rows stay where they are;
    they're context, not instructions.
    """
    if messages and isinstance(messages[0], SystemMessage) and messages[0].content == system_prompt:
        return messages
    return [SystemMessage(content=system_prompt), *messages]


def _error_envelope(code: str, message: str) -> str:
    """A tool-result error envelope, serialised properly.

    These used to be hand-assembled with an f-string, which produced invalid
    JSON the moment the interpolated text carried a newline or a quote — a
    pydantic ValidationError (the model got the arguments wrong) does both, so
    the one case the model most needs to read back was the one it got garbled.
    """
    envelope = {"ok": False, "error": {"code": code, "message": message}}
    return json.dumps(envelope, ensure_ascii=False)


#: Returned in place of a call that the per-response cap refused to run. Worded
#: so the model retries deliberately on its next turn instead of reading it as a
#: transient failure worth repeating verbatim.
_TOO_MANY_CALLS = _error_envelope(
    "TOO_MANY_TOOL_CALLS",
    "Not executed — this reply requested too many tools at once. Nothing ran for "
    "this call. Choose only the calls you actually need and make them in your "
    "next reply.",
)


def _call_key(call: dict[str, Any]) -> str:
    """Identity of a call for dedup: same tool + same arguments."""
    args = call.get("args")
    return json.dumps(
        [call.get("name"), args if isinstance(args, dict) else {}],
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def _is_confirmed_write(call: dict[str, Any]) -> bool:
    """True for the second leg of propose/confirm — the call the user already
    said yes to, carrying the proposal_token that commits the write."""
    args = call.get("args")
    return isinstance(args, dict) and bool(args.get("proposal_token"))


def _select_calls(calls: list[dict[str, Any]], limit: int) -> set[str]:
    """The distinct calls allowed to execute from one model response.

    Dedup first (collapsing repeats is free and loses nothing), then cap what is
    left. Confirmed writes are never the ones dropped: the model re-issues the
    write tool with its proposal_token once the user clicks confirm, and cutting
    that call surfaces as "I confirmed but nothing saved" — worse than the
    runaway it would be guarding against. So the cap eats into reads only, and a
    response somehow full of confirmed writes is allowed past the limit.
    """
    unique: list[str] = []
    writes: set[str] = set()
    for call in calls:
        key = _call_key(call)
        if key not in unique:
            unique.append(key)
        if _is_confirmed_write(call):
            writes.add(key)
    if len(unique) <= limit:
        return set(unique)

    selected = set(writes)
    for key in unique:
        if len(selected) >= limit:
            break
        selected.add(key)
    return selected


def _is_proposal(content: Any) -> bool:
    """True if a tool-result envelope is a propose (dry_run) response carrying a
    proposal_token to confirm. Tolerant of non-JSON / unexpected shapes."""
    try:
        env = json.loads(content) if isinstance(content, str) else content
    except (TypeError, ValueError):
        return False
    if not isinstance(env, dict) or not env.get("ok"):
        return False
    data = env.get("data") or {}
    return bool(isinstance(data, dict) and data.get("requires_confirm") and data.get("proposal_token"))


def history_to_messages(rows: list[dict[str, Any]]) -> list[BaseMessage]:
    """Convert persisted `for_llm` rows into LangChain messages for replay.

    Persisted `content` shape:
      user/system/assistant(text)  -> {"text": "..."}
      user with attachments        -> {"text": "...", "attachments": [...], "blocks": [...]}
      assistant(tool_calls)        -> {"text": "...", "tool_calls": [...]}
      tool                         -> {"tool_call_id", "name", "result"}

    `blocks` is not persisted — attachments.hydrate_history adds it just before
    replay, carrying multimodal content (document text, image data) rebuilt from
    the attachment store.

    The OpenAI/DeepSeek API requires every assistant tool_call to be followed by
    its matching tool message. Summarization can drop one side; to stay valid we
    degrade an assistant tool_call to plain text when its tool response isn't in
    this window, and we drop orphan tool messages.
    """
    answered: set[str] = set()
    for m in rows:
        if m.get("role") == "tool":
            tcid = (m.get("content") or {}).get("tool_call_id")
            if tcid:
                answered.add(tcid)

    out: list[BaseMessage] = []
    emitted_calls: set[str] = set()
    for m in rows:
        role = m.get("role")
        content = m.get("content") or {}
        text = content.get("text", "") or ""

        if role == "user":
            out.append(HumanMessage(content=content.get("blocks") or text))
        elif role == "system":
            out.append(SystemMessage(content=text))
        elif role == "assistant":
            raw_calls = content.get("tool_calls") or []
            valid = [c for c in raw_calls if c.get("id") in answered]
            if valid:
                out.append(
                    AIMessage(
                        content=text,
                        tool_calls=[
                            {
                                "id": c["id"],
                                "name": c["name"],
                                # An empty {} round-trips through JSON as [];
                                # AIMessage requires args to be a dict.
                                "args": c["args"] if isinstance(c.get("args"), dict) else {},
                            }
                            for c in valid
                        ],
                    )
                )
                emitted_calls.update(c["id"] for c in valid)
            else:
                out.append(AIMessage(content=text))
        elif role == "tool":
            tcid = content.get("tool_call_id")
            if tcid and tcid in emitted_calls:
                result = content.get("result")
                out.append(
                    ToolMessage(
                        content=json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result,
                        tool_call_id=tcid,
                    )
                )
            # else: orphan tool message -> drop
    return out
