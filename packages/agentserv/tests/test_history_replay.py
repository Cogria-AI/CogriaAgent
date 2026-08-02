"""history_to_messages: valid tool-call pairing + orphan degradation."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from cogria_agent.graph import history_to_messages


def test_plain_turns_roundtrip():
    rows = [
        {"role": "system", "content": {"text": "sys"}},
        {"role": "user", "content": {"text": "hi"}},
        {"role": "assistant", "content": {"text": "hello"}},
    ]
    msgs = history_to_messages(rows)
    assert [type(m) for m in msgs] == [SystemMessage, HumanMessage, AIMessage]
    assert msgs[2].content == "hello"


def test_paired_tool_call_replays():
    rows = [
        {"role": "user", "content": {"text": "do it"}},
        {"role": "assistant", "content": {"text": "", "tool_calls": [{"id": "c1", "name": "act", "args": {"x": 1}}]}},
        {"role": "tool", "content": {"tool_call_id": "c1", "name": "act", "result": {"ok": True}}},
    ]
    msgs = history_to_messages(rows)
    assert isinstance(msgs[1], AIMessage) and msgs[1].tool_calls[0]["id"] == "c1"
    assert isinstance(msgs[2], ToolMessage) and msgs[2].tool_call_id == "c1"


def test_orphan_assistant_call_degrades_to_text():
    # assistant tool_call whose tool response is missing (e.g. summarized away)
    rows = [
        {"role": "assistant", "content": {"text": "thinking", "tool_calls": [{"id": "c9", "name": "act", "args": {}}]}},
    ]
    msgs = history_to_messages(rows)
    assert isinstance(msgs[0], AIMessage)
    assert not msgs[0].tool_calls  # degraded
    assert msgs[0].content == "thinking"


def test_orphan_tool_message_dropped():
    rows = [
        {"role": "user", "content": {"text": "hi"}},
        {"role": "tool", "content": {"tool_call_id": "nope", "name": "act", "result": {}}},
    ]
    msgs = history_to_messages(rows)
    assert [type(m) for m in msgs] == [HumanMessage]


def test_empty_args_coerced_to_dict():
    # DB JSON round-trips empty {} as []; AIMessage requires args to be a dict
    rows = [
        {"role": "assistant", "content": {"text": "", "tool_calls": [{"id": "c1", "name": "act", "args": []}]}},
        {"role": "tool", "content": {"tool_call_id": "c1", "name": "act", "result": 1}},
    ]
    msgs = history_to_messages(rows)
    assert msgs[0].tool_calls[0]["args"] == {}


def test_hydrated_attachment_blocks_replay_as_multimodal_content():
    """attachments.hydrate_history adds `blocks`; replay must hand them to the
    model as-is rather than falling back to the bare text."""
    rows = [
        {
            "role": "user",
            "content": {
                "text": "what is in this?",
                "attachments": [{"id": "att_1", "name": "a.pdf"}],
                "blocks": [
                    {"type": "text", "text": "what is in this?\n\n<attachment id=\"att_1\">body</attachment>"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            },
        }
    ]
    msgs = history_to_messages(rows)
    assert len(msgs) == 1 and isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[0].content, list)
    assert "<attachment" in msgs[0].content[0]["text"]
    assert msgs[0].content[1]["type"] == "image_url"


def test_user_row_without_blocks_still_replays_as_plain_text():
    msgs = history_to_messages([{"role": "user", "content": {"text": "plain"}}])
    assert msgs[0].content == "plain"
