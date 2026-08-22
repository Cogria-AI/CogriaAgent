"""Tool-result pruning: bounded, idempotent, and safe for propose/confirm."""

from __future__ import annotations

import json

import pytest
from cogria_agent.config import PruneConfig
from cogria_agent.prune import TRUNCATION_KEY, prune_result


def _rows(n: int, width: int = 200) -> list[dict]:
    return [{"id": i, "note": "x" * width} for i in range(n)]


def _size(payload) -> int:
    if isinstance(payload, str):
        return len(payload)
    return len(json.dumps(payload, ensure_ascii=False))


def test_small_result_is_left_alone():
    assert prune_result({"ok": True, "data": {"items": _rows(2)}}) is None


def test_oversized_list_is_truncated_and_marked():
    payload = {"ok": True, "data": {"items": _rows(500)}}
    pruned = prune_result(payload)
    assert pruned is not None
    assert pruned["ok"] is True  # the envelope's own keys survive
    assert len(pruned["data"]["items"]) < 500
    # The model must be able to see it received a slice, not the whole set.
    assert pruned["data"][TRUNCATION_KEY]["omitted"] > 0


def test_pruned_result_stays_valid_json():
    """`graph._is_proposal` and the model both parse this back; a naive
    character cut through the middle of an object would break both."""
    pruned = prune_result({"ok": True, "data": {"items": _rows(500)}})
    json.loads(json.dumps(pruned, ensure_ascii=False))


def test_result_is_bounded_by_the_threshold():
    pruned = prune_result({"ok": True, "data": {"items": _rows(500)}}, threshold_chars=4_000)
    assert _size(pruned) <= 4_000


def test_pruning_is_idempotent():
    """A pruned result is under the threshold, so a second pass finds nothing.
    Without this, every turn would rewrite the same row again."""
    once = prune_result({"ok": True, "data": {"items": _rows(500)}})
    assert prune_result(once) is None


def test_proposal_envelope_is_never_touched():
    """The token has to come back verbatim. A truncation that split one would
    surface to the user as "I confirmed and nothing happened"."""
    payload = {
        "ok": True,
        "data": {
            "requires_confirm": True,
            "proposal_token": "tok_" + "a" * 64,
            "summary": "Delete " + "everything " * 2_000,
        },
    }
    assert _size(payload) > 8_192  # it IS oversized
    assert prune_result(payload) is None  # and is still exempt


def test_confirmed_write_envelope_is_also_exempt():
    payload = {"ok": True, "data": {"proposal_token": "tok_x", "note": "y" * 20_000}}
    assert prune_result(payload) is None


def test_oversized_plain_string_is_cut_head_and_tail():
    text = "START" + ("m" * 40_000) + "END"
    pruned = prune_result(text)
    assert pruned.startswith("START")
    assert pruned.endswith("END")
    assert "omitted" in pruned
    assert len(pruned) < len(text)


def test_result_with_no_list_falls_back_to_a_text_cut():
    payload = {"ok": True, "data": {"blob": "z" * 40_000}}
    pruned = prune_result(payload)
    assert pruned is not None
    assert _size(pruned) <= 8_192


def test_huge_rows_shrink_the_kept_count():
    """When the rows themselves are oversized, keeping twenty of them still
    blows the budget — the retry ladder has to keep fewer."""
    payload = {"ok": True, "data": {"items": _rows(50, width=4_000)}}
    pruned = prune_result(payload)
    assert _size(pruned) <= 8_192


def test_config_rejects_a_budget_that_could_not_shrink_anything():
    with pytest.raises(ValueError, match="less than threshold_chars"):
        PruneConfig(threshold_chars=1_000, head_chars=900, tail_chars=200).validated()


def test_default_config_is_valid():
    assert PruneConfig().validated().threshold_chars == 8_192


@pytest.mark.asyncio
async def test_persisted_turn_stores_the_pruned_copy():
    """End-to-end: the bound is applied where it matters — on the copy that
    every LATER turn will resend."""
    from cogria_agent.config import PruneConfig
    from cogria_agent.inmemory import InMemoryConversationBackend
    from cogria_agent.server import _persist_turn

    backend = InMemoryConversationBackend()
    cid = await backend.create_conversation(first_message="list them", model="m")
    huge = {"ok": True, "data": {"items": _rows(500)}}

    await _persist_turn(
        backend=backend,
        conversation_id=cid,
        acc={
            "assistant_text": "here you go",
            "pending_calls": {"c1": {"name": "list_orders", "args": {}}},
            "tool_results": [
                {"tool_call_id": "c1", "name": "list_orders",
                 "content": json.dumps(huge, ensure_ascii=False)}
            ],
            "input_tokens": 1,
            "output_tokens": 1,
            "finish_reason": "stop",
            "stream_error": None,
        },
        artifact_names=set(),
        model="m",
        prune_config=PruneConfig(),
    )

    rows = await backend.fetch_messages_full(cid)
    stored = next(r for r in rows if r["role"] == "tool")["content"]["result"]
    assert _size(stored) <= 8_192
    # The envelope's own keys survive; the truncation marker sits beside the
    # shortened list so the model can see it received a slice.
    assert stored["ok"] is True
    assert stored["data"][TRUNCATION_KEY]["omitted"] > 0


@pytest.mark.asyncio
async def test_persisting_without_a_prune_config_changes_nothing():
    """Pruning is opt-outable and must not be load-bearing for correctness."""
    from cogria_agent.inmemory import InMemoryConversationBackend
    from cogria_agent.server import _persist_turn

    backend = InMemoryConversationBackend()
    cid = await backend.create_conversation(first_message="list them", model="m")
    huge = {"ok": True, "data": {"items": _rows(500)}}

    await _persist_turn(
        backend=backend,
        conversation_id=cid,
        acc={
            "assistant_text": "",
            "pending_calls": {"c1": {"name": "list_orders", "args": {}}},
            "tool_results": [
                {"tool_call_id": "c1", "name": "list_orders",
                 "content": json.dumps(huge, ensure_ascii=False)}
            ],
            "input_tokens": 0,
            "output_tokens": 0,
            "finish_reason": "tool_calls",
            "stream_error": None,
        },
        artifact_names=set(),
        model="m",
    )
    rows = await backend.fetch_messages_full(cid)
    stored = next(r for r in rows if r["role"] == "tool")["content"]["result"]
    assert len(stored["data"]["items"]) == 500


def test_a_list_valued_data_stays_a_list():
    """Shrinking must not change the shape the action documented: turning a
    list-valued `data` into an object to make room for the marker would break
    the contract the model was taught to read."""
    pruned = prune_result({"ok": True, "data": _rows(500)})
    assert isinstance(pruned["data"], list)
    assert len(pruned["data"]) < 500
    # The marker moves up to the envelope, where there IS room for it.
    assert pruned[TRUNCATION_KEY]["omitted"] > 0
    assert _size(pruned) <= 8_192
    assert prune_result(pruned) is None  # still idempotent
