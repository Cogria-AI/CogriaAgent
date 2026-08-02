"""Summarization over the in-memory backend with a fake summary LLM."""

from __future__ import annotations

import pytest

from cogria_agent.config import SummarizerConfig
from cogria_agent.inmemory import InMemoryConversationBackend
from cogria_agent.summarizer import maybe_summarize


class _FakeResp:
    content = "RUNNING SUMMARY"


class _FakeSummaryLLM:
    async def ainvoke(self, messages):
        return _FakeResp()


class _FakeLLMFactory:
    def summary_llm(self):
        return _FakeSummaryLLM()


@pytest.mark.asyncio
async def test_no_summary_under_threshold():
    backend = InMemoryConversationBackend()
    cid = await backend.create_conversation(first_message="hi", model="m")
    cfg = SummarizerConfig(message_threshold=50, keep_recent=20)
    assert await maybe_summarize(backend=backend, llm_factory=_FakeLLMFactory(), config=cfg, conversation_id=cid) is False


@pytest.mark.asyncio
async def test_summary_folds_old_messages():
    backend = InMemoryConversationBackend()
    cid = await backend.create_conversation(first_message="msg0", model="m")
    # push the count over a small threshold
    extra = [{"role": "assistant", "content": {"text": f"msg{i}"}} for i in range(1, 12)]
    await backend.append_messages(cid, messages=extra, usage=None, model="m")
    cfg = SummarizerConfig(message_threshold=5, keep_recent=3, token_threshold=10_000)

    did = await maybe_summarize(backend=backend, llm_factory=_FakeLLMFactory(), config=cfg, conversation_id=cid)
    assert did is True

    replay = await backend.fetch_history(cid, for_llm=True)
    # summary head + the kept-recent tail
    assert replay[0]["role"] == "system"
    assert replay[0]["content"]["text"] == "RUNNING SUMMARY"
    assert len(replay) == 1 + 3  # summary + keep_recent


@pytest.mark.asyncio
async def test_render_keeps_attachment_names():
    """A folded turn stops replaying its files, so the transcript handed to the
    summary model has to name them or the assistant forgets they existed."""
    from cogria_agent.summarizer import _render

    rendered = _render(
        [
            {
                "role": "user",
                "content": {
                    "text": "check this",
                    "attachments": [{"id": "att_1", "name": "menu.pdf"}, {"id": "att_2", "name": "q3.docx"}],
                },
            }
        ]
    )
    assert "check this" in rendered
    assert "menu.pdf" in rendered and "q3.docx" in rendered
