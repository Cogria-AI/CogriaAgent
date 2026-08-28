"""Compaction over the in-memory backend with a fake summary LLM.

The first two tests are the regression anchors for the bug this module was
rewritten to fix: the trigger used to compare LIFETIME token usage against a
context budget. Because every turn resends the whole history, lifetime usage
grows quadratically and says nothing about whether the next request fits — so
the old implementation fired on conversations that were nowhere near full and
stayed quiet on ones that were. Both cases are asserted below and both fail
against the old behaviour.
"""

from __future__ import annotations

import pytest
from cogria_agent.config import SummarizerConfig
from cogria_agent.inmemory import InMemoryConversationBackend
from cogria_agent.summarizer import (
    CHECKPOINT_OPEN,
    _align_cutoff,
    compact,
    maybe_summarize,
)


class _FakeResp:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeSummaryLLM:
    def __init__(self, recorder: dict) -> None:
        self._recorder = recorder

    def bind_tools(self, tools):
        self._recorder["bound_tools"] = [getattr(t, "name", t) for t in tools]
        return self

    async def ainvoke(self, messages):
        self._recorder["calls"] = self._recorder.get("calls", 0) + 1
        self._recorder["messages"] = messages
        return _FakeResp("RUNNING SUMMARY")


class _FakeLLMFactory:
    def __init__(self) -> None:
        self.recorder: dict = {}

    def summary_llm(self):
        return _FakeSummaryLLM(self.recorder)


def _cfg(**kw) -> SummarizerConfig:
    base = {"context_window": 1_000, "threshold_ratio": 0.7, "retain_ratio": 0.2}
    base.update(kw)
    return SummarizerConfig(**base)


async def _seed(backend, cid, count, *, text="x"):
    await backend.append_messages(
        cid,
        messages=[{"role": "assistant", "content": {"text": f"{text}{i}"}} for i in range(count)],
        usage=None,
        model="m",
    )


async def _seed_heavy(backend, cid, count=12, words=500):
    """A conversation long enough for a verbatim tail to leave something to fold.

    `keep_recent` is a floor in messages, so a four-message conversation has
    nothing compactable however large those messages are — which is correct
    (folding everything leaves nothing to continue from) but makes for a
    misleading fixture.
    """
    await backend.append_messages(
        cid,
        messages=[
            {"role": "assistant", "content": {"text": f"reply{i} " + "word " * words}}
            for i in range(count)
        ],
        usage=None,
        model="m",
    )


@pytest.mark.asyncio
async def test_many_cheap_turns_do_not_trigger():
    """Lifetime usage far past the old threshold, but a tiny prompt.

    Twenty short exchanges bill a large cumulative total (each turn resends
    everything before it) while the next request is a few hundred tokens. The
    old judgement compacted here; the correct one leaves it alone.
    """
    backend = InMemoryConversationBackend()
    cid = await backend.create_conversation(first_message="hi", model="m")
    await _seed(backend, cid, 20, text="short reply ")
    # Bill a lifetime total that would have blown the old 16k threshold apart.
    await backend.append_messages(
        cid, messages=[], usage={"input_tokens": 400_000, "output_tokens": 40_000}, model="m"
    )

    factory = _FakeLLMFactory()
    did = await maybe_summarize(
        backend=backend, llm_factory=factory, config=_cfg(), conversation_id=cid
    )
    assert did is False
    assert factory.recorder.get("calls") is None


@pytest.mark.asyncio
async def test_few_expensive_turns_do_trigger():
    """A handful of huge messages: low message count, low lifetime usage, and a
    prompt that will not fit. The old judgement stayed quiet here."""
    backend = InMemoryConversationBackend()
    cid = await backend.create_conversation(first_message="hi", model="m")
    await _seed_heavy(backend, cid)
    await backend.append_messages(
        cid, messages=[], usage={"input_tokens": 10, "output_tokens": 10}, model="m"
    )

    factory = _FakeLLMFactory()
    did = await maybe_summarize(
        backend=backend, llm_factory=factory, config=_cfg(), conversation_id=cid
    )
    assert did is True

    replay = await backend.fetch_history(cid, for_llm=True)
    assert replay[0]["role"] == "user"
    assert CHECKPOINT_OPEN in replay[0]["content"]["text"]
    assert "RUNNING SUMMARY" in replay[0]["content"]["text"]


@pytest.mark.asyncio
async def test_summary_call_replays_the_conversation_prefix():
    """The summarization request must be a genuine prefix of the conversation's
    own request — same system prompt, same tools — or the provider's prompt
    cache is invalidated and the call is billed at full price."""

    class _Tool:
        name = "list_orders"

    backend = InMemoryConversationBackend()
    cid = await backend.create_conversation(first_message="hi", model="m")
    await _seed_heavy(backend, cid)

    factory = _FakeLLMFactory()
    assert await maybe_summarize(
        backend=backend,
        llm_factory=factory,
        config=_cfg(),
        conversation_id=cid,
        system_prompt="YOU ARE THE OPERATING PROMPT",
        tools=[_Tool()],
    )

    sent = factory.recorder["messages"]
    assert sent[0].content == "YOU ARE THE OPERATING PROMPT"
    assert factory.recorder["bound_tools"] == ["list_orders"]
    # The instruction rides last, after the replayed history.
    assert "checkpoint" in sent[-1].content.lower()


@pytest.mark.asyncio
async def test_second_compaction_is_incremental():
    """Only the span added since the last checkpoint is re-read.

    Re-summarizing from message zero every time made each compaction cost more
    than the last, to reproduce what the previous checkpoint already held.
    """
    backend = InMemoryConversationBackend()
    cid = await backend.create_conversation(first_message="hi", model="m")
    await _seed_heavy(backend, cid)
    factory = _FakeLLMFactory()
    cfg = _cfg()
    assert await maybe_summarize(
        backend=backend, llm_factory=factory, config=cfg, conversation_id=cid
    )
    first_len = len(factory.recorder["messages"])

    # Add the same amount again and compact a second time.
    await _seed_heavy(backend, cid)
    assert await maybe_summarize(
        backend=backend, llm_factory=factory, config=cfg, conversation_id=cid
    )
    second = factory.recorder["messages"]

    # The prior checkpoint leads the span, so the second call reads roughly the
    # same amount as the first — not twice as much.
    assert len(second) <= first_len + 1
    # And it is handed the previous checkpoint so it merges rather than nests.
    assert any("RUNNING SUMMARY" in str(getattr(m, "content", "")) for m in second)


@pytest.mark.asyncio
async def test_concurrent_compaction_runs_once():
    """A fast follow-up turn must not start a second compaction over the same
    span: same work, same cost, and one result overwrites the other."""
    import asyncio

    backend = InMemoryConversationBackend()
    cid = await backend.create_conversation(first_message="hi", model="m")
    await _seed_heavy(backend, cid)

    factory = _FakeLLMFactory()
    results = await asyncio.gather(
        *[
            maybe_summarize(
                backend=backend, llm_factory=factory, config=_cfg(), conversation_id=cid
            )
            for _ in range(4)
        ]
    )
    assert sum(1 for r in results if r) == 1
    assert factory.recorder["calls"] == 1


@pytest.mark.asyncio
async def test_summary_input_is_capped_rather_than_failing():
    """An over-long span is trimmed from the front, not abandoned.

    Losing the oldest slice beats losing compaction itself: an exception here
    used to be swallowed, so the conversation kept growing with nothing holding
    it back and the only symptom was a warning in the log.
    """
    backend = InMemoryConversationBackend()
    cid = await backend.create_conversation(first_message="hi", model="m")
    await backend.append_messages(
        cid,
        messages=[{"role": "assistant", "content": {"text": "word " * 5_000}} for _ in range(20)],
        usage=None,
        model="m",
    )

    factory = _FakeLLMFactory()
    assert await maybe_summarize(
        backend=backend,
        llm_factory=factory,
        config=_cfg(max_summary_input_ratio=0.3),
        conversation_id=cid,
    )
    # Whatever was sent stayed inside the cap rather than raising.
    assert factory.recorder["calls"] == 1
    assert "not shown" in factory.recorder["messages"][-1].content


@pytest.mark.asyncio
async def test_failure_is_swallowed_but_counted(caplog):
    class _Boom:
        def summary_llm(self):
            raise RuntimeError("summary model unreachable")

    backend = InMemoryConversationBackend()
    cid = await backend.create_conversation(first_message="hi", model="m")
    await _seed_heavy(backend, cid)

    cfg = _cfg()
    assert await maybe_summarize(
        backend=backend, llm_factory=_Boom(), config=cfg, conversation_id=cid
    ) is False
    # The second consecutive failure escalates: a conversation that can no
    # longer compact is growing with nothing to stop it.
    with caplog.at_level("ERROR"):
        assert await maybe_summarize(
            backend=backend, llm_factory=_Boom(), config=cfg, conversation_id=cid
        ) is False
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_cutoff_never_splits_a_tool_pair():
    """Folding an assistant tool call away from its result leaves the model
    reading an announcement whose answer never arrives."""
    rows = [
        {"role": "user", "content": {"text": "go"}},
        {
            "role": "assistant",
            "content": {"text": "", "tool_calls": [{"id": "c1", "name": "t", "args": {}}]},
        },
        {"role": "tool", "content": {"tool_call_id": "c1", "name": "t", "result": {"ok": True}}},
        {"role": "assistant", "content": {"text": "done"}},
    ]
    # A cut between the call and its result is pulled back in front of the call.
    assert _align_cutoff(rows, 2) == 1
    # A cut that keeps the pair together is left alone.
    assert _align_cutoff(rows, 3) == 3


def test_cutoff_alignment_spans_a_multi_call_round():
    """One assistant message can open several calls answered over several rows;
    every cut inside that whole stretch has to be rejected, not just the one
    immediately after the announcement."""
    rows = [
        {"role": "user", "content": {"text": "go"}},
        {
            "role": "assistant",
            "content": {
                "text": "",
                "tool_calls": [
                    {"id": "c1", "name": "t", "args": {}},
                    {"id": "c2", "name": "t", "args": {}},
                    {"id": "c3", "name": "t", "args": {}},
                ],
            },
        },
        {"role": "tool", "content": {"tool_call_id": "c1", "name": "t", "result": {}}},
        {"role": "tool", "content": {"tool_call_id": "c2", "name": "t", "result": {}}},
        {"role": "tool", "content": {"tool_call_id": "c3", "name": "t", "result": {}}},
        {"role": "assistant", "content": {"text": "done"}},
    ]
    # Every cut from 2 through 4 lands inside the round and is pulled back.
    for candidate in (2, 3, 4):
        assert _align_cutoff(rows, candidate) == 1
    # Just past the last result is a clean boundary.
    assert _align_cutoff(rows, 5) == 5


def test_cutoff_alignment_tolerates_an_unanswered_call():
    """A call with no result in the span (a cancelled turn, a crash) must not
    forbid every cut after it — that would stop compaction entirely."""
    rows = [
        {"role": "user", "content": {"text": "go"}},
        {
            "role": "assistant",
            "content": {"text": "", "tool_calls": [{"id": "gone", "name": "t", "args": {}}]},
        },
        {"role": "user", "content": {"text": "never mind"}},
        {"role": "assistant", "content": {"text": "ok"}},
    ]
    assert _align_cutoff(rows, 3) == 3


@pytest.mark.asyncio
async def test_force_compacts_below_threshold():
    """The recovery path after a provider has already rejected a request: it
    must compact even though pressure alone would not justify it."""
    backend = InMemoryConversationBackend()
    cid = await backend.create_conversation(first_message="hi", model="m")
    await _seed(backend, cid, 8)

    factory = _FakeLLMFactory()
    cfg = _cfg(context_window=1_000_000)  # nowhere near pressure
    assert await maybe_summarize(
        backend=backend, llm_factory=factory, config=cfg, conversation_id=cid
    ) is False
    assert await compact(
        backend=backend, llm_factory=factory, config=cfg, conversation_id=cid, force=True
    ) is True


@pytest.mark.asyncio
async def test_render_keeps_attachment_names():
    """A folded turn stops replaying its files, so the flat transcript handed to
    the summary model has to name them or the assistant forgets they existed."""
    from cogria_agent.summarizer import _render

    rendered = _render(
        [
            {
                "role": "user",
                "content": {
                    "text": "check this",
                    "attachments": [
                        {"id": "att_1", "name": "menu.pdf"},
                        {"id": "att_2", "name": "q3.docx"},
                    ],
                },
            }
        ]
    )
    assert "check this" in rendered
    assert "menu.pdf" in rendered and "q3.docx" in rendered


@pytest.mark.asyncio
async def test_retention_prices_attachments_like_pressure_does():
    """The tail budget and the pressure check must agree on what a photo costs.

    Making pressure attachment-aware while retention stayed blind let a
    photo-heavy tail keep several times its budget: the tail reads as its
    filenames, compaction under-delivers, and the retry loop burns its attempts
    on a conversation it never actually shrank.
    """
    from cogria_agent.config import AttachmentsConfig
    from cogria_agent.summarizer import _select_cutoff

    photo = {
        "role": "user",
        "content": {"text": "午饭", "attachments": [{"id": "a", "name": "l.jpg", "kind": "image"}]},
    }
    rows = [dict(photo) for _ in range(12)]
    cfg = _cfg(retain_ratio=0.2, keep_recent=1)  # 200 tokens of tail

    blind = _select_cutoff(rows, cfg)
    aware = _select_cutoff(rows, cfg, AttachmentsConfig(vision_model="v"))

    # Blind to the pictures, the tail looks nearly free and keeps walking back.
    # Priced, one photo turn alone already fills the budget.
    assert aware > blind
    assert aware == len(rows) - 1


@pytest.mark.asyncio
async def test_prefix_reuse_is_skipped_for_a_different_summary_model():
    """Replaying the prefix pays off only against the model that warmed the
    cache. Routed to a cheaper model it is a pure loss: a full structured
    replay instead of a flat transcript, billed at list price."""

    class _Cheap(_FakeLLMFactory):
        def model_name(self):
            return "big-chat-model"

        def summary_llm(self):
            llm = _FakeSummaryLLM(self.recorder)
            llm.model_name = "cheap-summary-model"
            return llm

    backend = InMemoryConversationBackend()
    cid = await backend.create_conversation(first_message="hi", model="m")
    await _seed_heavy(backend, cid)

    factory = _Cheap()
    assert await maybe_summarize(
        backend=backend,
        llm_factory=factory,
        config=_cfg(),
        conversation_id=cid,
        system_prompt="OPERATING PROMPT",
    )
    sent = factory.recorder["messages"]
    # The flat form: an instruction and a transcript, not a replayed prefix.
    assert len(sent) == 2
    assert "OPERATING PROMPT" not in str(sent[0].content)


@pytest.mark.asyncio
async def test_prefix_reuse_survives_a_factory_it_cannot_introspect():
    """An injected factory may route however it likes. Unable to tell, honour
    what the caller configured rather than silently downgrading it."""
    backend = InMemoryConversationBackend()
    cid = await backend.create_conversation(first_message="hi", model="m")
    await _seed_heavy(backend, cid)

    factory = _FakeLLMFactory()  # no model_name() at all
    assert await maybe_summarize(
        backend=backend,
        llm_factory=factory,
        config=_cfg(),
        conversation_id=cid,
        system_prompt="OPERATING PROMPT",
    )
    assert factory.recorder["messages"][0].content == "OPERATING PROMPT"
