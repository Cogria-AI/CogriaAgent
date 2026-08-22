"""Token estimation over persisted message rows."""

from __future__ import annotations

from cogria_agent import estimate
from cogria_agent.estimate import (
    _heuristic,
    estimate_message,
    estimate_messages,
    estimate_text,
    reset_encoder_cache,
)


def test_empty_text_is_free():
    assert estimate_text("") == 0


def test_longer_text_costs_more():
    assert estimate_text("word " * 100) > estimate_text("word " * 10)


def test_heuristic_does_not_undercount_cjk():
    """`len/4` is the usual rule of thumb and it is badly wrong for Chinese: a
    character is closer to one token than to a quarter of one. Under-counting
    here is what would let a Chinese conversation sail past the threshold."""
    chinese = "订单已经确认请尽快安排配送谢谢"
    latin = "x" * len(chinese)
    assert _heuristic(chinese) > _heuristic(latin) * 3


def test_falls_back_when_tiktoken_is_unavailable(monkeypatch):
    """A deployment with no egress cannot fetch a BPE table. That must degrade
    to the heuristic rather than raising on every estimate."""
    reset_encoder_cache()
    monkeypatch.setattr(estimate, "_get_encoder", lambda: None)
    assert estimate_text("hello world") == _heuristic("hello world")
    reset_encoder_cache()


def test_tool_result_is_priced():
    """A tool result is the biggest thing history carries and the easiest to
    forget: it lives under `result`, not `text`."""
    bare = {"role": "tool", "content": {"tool_call_id": "c1", "name": "t", "result": {}}}
    heavy = {
        "role": "tool",
        "content": {
            "tool_call_id": "c1",
            "name": "t",
            "result": {"data": {"items": [{"name": f"row {i}"} for i in range(200)]}},
        },
    }
    assert estimate_message(heavy) > estimate_message(bare) * 10


def test_tool_call_arguments_are_priced():
    row = {
        "role": "assistant",
        "content": {
            "text": "",
            "tool_calls": [{"id": "c1", "name": "search", "args": {"q": "x" * 400}}],
        },
    }
    assert estimate_message(row) > 50


def test_images_are_priced_over_their_placeholder_text():
    """An image costs hundreds of tokens while its row's text is empty. Pricing
    the text alone would report a nearly free message."""
    text_only = {"role": "user", "content": {"text": "look"}}
    with_image = {
        "role": "user",
        "content": {
            "text": "look",
            "blocks": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        },
    }
    assert estimate_message(with_image) > estimate_message(text_only) + 500


def test_messages_sum():
    rows = [{"role": "user", "content": {"text": "hello there"}} for _ in range(5)]
    assert estimate_messages(rows) == sum(estimate_message(r) for r in rows)


def test_non_dict_content_does_not_raise():
    assert estimate_message({"role": "user", "content": "plain string"}) > 0
    assert estimate_message({}) > 0
