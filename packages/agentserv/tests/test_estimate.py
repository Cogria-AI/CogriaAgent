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


# --- attachment content: priced from config, because the rows don't carry it ---

from cogria_agent.config import AttachmentsConfig  # noqa: E402


def _photo_turn(n: int = 1) -> dict:
    return {
        "role": "user",
        "content": {
            "text": "这是我今天的午饭",
            "attachments": [
                {"id": f"att_{i}", "name": f"lunch{i}.jpg", "mime": "image/jpeg", "kind": "image"}
                for i in range(n)
            ],
        },
    }


def _doc_turn(chars: int = 20_000) -> dict:
    return {
        "role": "user",
        "content": {
            "text": "看看这份报表",
            "attachments": [
                {
                    "id": "att_d",
                    "name": "q3.xlsx",
                    "mime": "application/vnd.ms-excel",
                    "kind": "text",
                }
            ],
        },
    }


VISION = AttachmentsConfig(vision_model="gpt-4o-mini")


def test_a_photo_turn_is_not_priced_as_its_filename():
    """The defect this exists to close: a stored row names the file, the request
    carries the picture. Pricing the row alone reads ~15 tokens for something
    that costs ~800, and the error runs the unsafe way — a full conversation
    looks roomy, so compaction waits."""
    rows = [_photo_turn()]
    naive = estimate_messages(rows)
    priced = estimate_messages(rows, attachments=VISION)
    assert priced - naive >= 700


def test_image_pricing_follows_image_history_turns():
    """Only the most recent image-bearing turns are actually re-sent; older ones
    degrade to a placeholder. Pricing every one of them would swap an
    under-estimate for an over-estimate."""
    rows = [_photo_turn(), _photo_turn(), _photo_turn(), _photo_turn()]
    def priced(turns: int) -> int:
        cfg = AttachmentsConfig(vision_model="v", image_history_turns=turns)
        return estimate_messages(rows, attachments=cfg)

    one, three = priced(1), priced(3)
    assert three - one >= 1_400  # two extra turns of pictures


def test_several_images_in_one_turn_all_count():
    single = estimate_messages([_photo_turn(1)], attachments=VISION)
    triple = estimate_messages([_photo_turn(3)], attachments=VISION)
    assert triple - single >= 1_400


def test_images_are_free_without_a_vision_model():
    """Image uploads are refused outright when no vision model is configured, so
    no image ever reaches the request."""
    rows = [_photo_turn()]
    assert estimate_messages(rows, attachments=AttachmentsConfig()) == estimate_messages(rows)


def test_document_text_is_priced_from_its_allowance():
    rows = [_doc_turn()]
    naive = estimate_messages(rows)
    priced = estimate_messages(rows, attachments=AttachmentsConfig(max_chars_per_doc=20_000))
    # A Chinese conversation, so the 20 000-character allowance is worth far
    # more than the Latin rule of thumb would suggest.
    assert priced - naive > 10_000


def test_document_pricing_follows_the_conversation_language():
    """A character budget is not a token budget, and the exchange rate is the
    language. 40 000 characters is ~10 000 tokens of English and ~36 000 of
    Chinese; assuming either constant is wrong for half the deployments."""
    cfg = AttachmentsConfig(max_chars_per_doc=20_000, max_chars_total=20_000)
    latin = dict(_doc_turn())
    latin["content"] = {**latin["content"], "text": "please take a look at this quarterly report"}

    chinese_cost = estimate_messages([_doc_turn()], attachments=cfg)
    latin_cost = estimate_messages([latin], attachments=cfg)
    assert chinese_cost > latin_cost * 2


def test_document_pricing_respects_the_whole_request_budget():
    """`Budget` caps the whole request, not just each file: ten documents cannot
    inject ten times the per-doc allowance."""
    cfg = AttachmentsConfig(max_chars_per_doc=20_000, max_chars_total=40_000)
    two = estimate_messages([_doc_turn() for _ in range(2)], attachments=cfg)
    ten = estimate_messages([_doc_turn() for _ in range(10)], attachments=cfg)
    # The extra eight documents add their <attachment/> references and nothing
    # else — the injection budget was already spent by the first two.
    assert ten - two < 1_000


def test_hydrated_rows_are_not_double_counted():
    """A row that already carries `blocks` was priced from its real content;
    adding the modelled cost on top would count the same picture twice."""
    hydrated = {
        "role": "user",
        "content": {
            "text": "这是我今天的午饭",
            "attachments": [{"id": "a", "name": "l.jpg", "kind": "image"}],
            "blocks": [
                {"type": "text", "text": "这是我今天的午饭"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
            ],
        },
    }
    assert estimate_messages([hydrated], attachments=VISION) == estimate_messages([hydrated])


def test_omitting_the_config_changes_nothing_for_text_only_rows():
    """A deployment with uploads switched off must price exactly as before."""
    rows = [{"role": "user", "content": {"text": "hello"}} for _ in range(5)]
    assert estimate_messages(rows, attachments=VISION) == estimate_messages(rows)
