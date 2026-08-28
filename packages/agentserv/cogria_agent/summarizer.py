"""Context compaction: fold the older part of a conversation into a checkpoint.

Runs after a turn is persisted, and on demand when a request has already been
rejected for being too long. Business-neutral: the instruction comes from
config, so a project may add domain hints without touching this module.

What decides when it runs
-------------------------
The estimated size of the messages the *next* request will replay, against
`context_window × threshold_ratio`. Not lifetime token usage: every turn
resends the whole history, so cumulative usage grows quadratically and has no
relationship to whether the next request fits.

What it keeps
-------------
A verbatim tail worth `retain_ratio` of the window (at least `keep_recent`
messages), with the cut point moved back until it does not separate an
assistant's tool call from that call's result. `graph.history_to_messages` can
repair a split pair, but repairing means the model reads an assistant message
announcing a tool call whose result never arrives — better not to create one.

What it produces
----------------
One checkpoint message replacing the folded span, stored through
`save_summary`. It is a *user* message: the operating prompt is the only
system-role instruction the model should receive, and a checkpoint sitting in
that role reads as one. `SUMMARY_PREAMBLE` frames it as background instead.

Failures never break chat. They are counted, and a conversation that keeps
failing escalates from warning to error, because the failure mode this replaces
was silent: compaction stopped working and nothing said so until the provider
rejected an oversized request.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .config import SUMMARY_PREAMBLE, AttachmentsConfig, SummarizerConfig
from .estimate import estimate_message, estimate_messages, estimate_row_costs, estimate_text
from .graph import history_to_messages
from .protocols import ConversationBackend

logger = logging.getLogger("cogria.summarizer")

#: Conversations with a compaction in flight. Compaction is fired per turn and
#: is not instantaneous, so a fast follow-up can start a second one over the
#: same span: same work, same cost, and the later result may be overwritten by
#: the earlier one finishing second. Nothing is corrupted — each write is
#: self-consistent — but the second call is pure waste.
_in_flight: set[str] = set()

#: Consecutive failures per conversation, so a persistent problem escalates.
_failures: dict[str, int] = {}

#: Marks the checkpoint inside the stored message, so a later compaction can
#: recognise a previous checkpoint and merge rather than nest.
CHECKPOINT_OPEN = "<checkpoint>"
CHECKPOINT_CLOSE = "</checkpoint>"


def frame_summary(summary: str) -> str:
    """Wrap raw summary text in the durable checkpoint framing."""
    return f"{SUMMARY_PREAMBLE}\n\n{CHECKPOINT_OPEN}\n{summary.strip()}\n{CHECKPOINT_CLOSE}"


def _render(messages: list[dict[str, Any]]) -> str:
    """Flatten persisted rows into a plain transcript.

    Used only when prefix reuse is switched off; the default path replays
    structured messages instead. Kept because a project pointing
    `summary_model` at a different provider gains nothing from prefix reuse and
    may prefer the cheaper flat form.
    """
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content") or {}
        if isinstance(content, dict):
            text = content.get("text") or ""
            if not text and content.get("result") is not None:
                text = json.dumps(content["result"], ensure_ascii=False)[:500]
            # Once a turn is folded into the summary its attachments are no
            # longer replayed, so the file names have to survive here or the
            # assistant forgets a document was ever shared.
            names = [
                a.get("name")
                for a in (content.get("attachments") or [])
                if isinstance(a, dict) and a.get("name")
            ]
            if names:
                text = f"{text} [attached files: {', '.join(names)}]".strip()
        else:
            text = str(content)
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _forbidden_cutoffs(rows: list[dict[str, Any]]) -> list[bool]:
    """Which cut positions would separate a tool call from its result.

    `rows[:c]` is folded and `rows[c:]` replayed, so a call at index `a`
    answered at index `t` rules out every `c` in `a < c <= t` — those are the
    cuts that leave the call behind and its answer in front.

    Marked with a difference array so the whole set is one pass rather than one
    rescan per candidate; the folded span can be thousands of rows.
    """
    answer_index: dict[str, int] = {}
    for index, row in enumerate(rows):
        if row.get("role") == "tool":
            tcid = (row.get("content") or {}).get("tool_call_id")
            if tcid and tcid not in answer_index:
                answer_index[tcid] = index

    marks = [0] * (len(rows) + 2)
    for index, row in enumerate(rows):
        if row.get("role") != "assistant":
            continue
        for call in (row.get("content") or {}).get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            answered_at = answer_index.get(call.get("id"))
            # A result before its own call is malformed; ignore rather than
            # forbid a range that runs backwards.
            if answered_at is None or answered_at < index:
                continue
            marks[index + 1] += 1
            marks[answered_at + 1] -= 1

    forbidden: list[bool] = []
    running = 0
    for position in range(len(rows) + 1):
        running += marks[position]
        forbidden.append(running > 0)
    return forbidden


def _align_cutoff(rows: list[dict[str, Any]], cutoff: int) -> int:
    """Move `cutoff` back until it does not split a tool call from its result.

    `history_to_messages` can repair a split pair, but repairing means the model
    reads an assistant message announcing a call whose result never arrives —
    better not to create one. Walking backwards always terminates: 0 folds
    nothing and so splits nothing.
    """
    forbidden = _forbidden_cutoffs(rows)
    while cutoff > 0 and forbidden[cutoff]:
        cutoff -= 1
    return cutoff


def _select_cutoff(
    rows: list[dict[str, Any]],
    config: SummarizerConfig,
    attachments: AttachmentsConfig | None = None,
) -> int:
    """How many leading rows to fold, keeping a token-sized verbatim tail.

    Counting messages rather than tokens — the previous behaviour — measures
    the wrong thing in both directions: twenty short exchanges are nothing to
    keep, while twenty rows carrying three large tool results are most of the
    window.

    Retention is priced with the same `attachments` the pressure check uses.
    Sizing the tail without them while measuring pressure with them would let a
    photo-heavy tail keep several times its budget: the tail reads as cheap,
    compaction under-delivers, and the retry loop runs out of attempts on a
    conversation it never actually shrank.
    """
    if not rows:
        return 0

    costs = estimate_row_costs(rows, attachments=attachments)
    retain = config.retain_tokens
    accumulated = 0
    keep_from = len(rows)
    for index in range(len(rows) - 1, -1, -1):
        accumulated += costs[index]
        keep_from = index
        if accumulated >= retain:
            break

    # Honour the message floor, then align so no tool pair is split.
    keep_from = min(keep_from, max(0, len(rows) - config.keep_recent))
    return _align_cutoff(rows, keep_from)


def _cap_input(
    rows: list[dict[str, Any]], config: SummarizerConfig
) -> tuple[list[dict[str, Any]], int, bool]:
    """Trim the folded span from the front so one summarization call fits.

    Returns the rows to summarize, how many leading rows were dropped, and
    whether anything was dropped. Dropping from the front rather than failing is
    the point: an over-long span used to raise, get swallowed, and leave the
    conversation uncompacted forever — losing the oldest slice is strictly
    better than losing compaction itself.
    """
    budget = config.max_summary_input_tokens
    # Priced WITHOUT attachments on purpose: the summarization call replays the
    # stored rows through `history_to_messages`, which does not hydrate, so no
    # file content reaches it. Charging for content that will not be sent would
    # trim the span for no reason.
    #
    # Price each row ONCE. Re-estimating the shrinking suffix on every step
    # would tokenize the same text over and over — quadratic work on exactly the
    # path that runs when a conversation has already grown large.
    costs = [estimate_message(row) for row in rows]
    total = sum(costs)
    if total <= budget:
        return rows, 0, False

    start = 0
    while start < len(rows) - 1 and total > budget:
        total -= costs[start]
        start += 1
    return rows[start:], start, True


async def _summarize(
    *,
    llm_factory: Any,
    config: SummarizerConfig,
    rows: list[dict[str, Any]],
    system_prompt: str | None,
    tools: list[Any] | None,
    dropped_prefix: bool,
) -> str:
    """One summarization call. Returns the raw checkpoint text.

    The default path replays the conversation's own system prompt, tool schemas
    and messages, then appends the instruction as the final user message. That
    makes the call a genuine prefix of the request the conversation just sent,
    so a provider with prompt caching serves almost all of it from cache and
    charges only for the instruction. Omitting the tool schemas would break
    that: they are serialized ahead of the messages, so a mismatch there
    invalidates the prefix from its first token.
    """
    instruction = config.prompt
    if dropped_prefix:
        instruction = (
            "Note: the earliest part of this conversation is not shown; summarize "
            "what is visible and do not speculate about what came before.\n\n" + instruction
        )

    llm = llm_factory.summary_llm()

    if config.reuse_conversation_prefix:
        messages: list[Any] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.extend(history_to_messages(rows))
        messages.append(HumanMessage(content=instruction))
        # Bound but never called: the instruction forbids it. They are here so
        # the serialized prefix matches the conversation's own requests.
        # `summary_llm()` comes from an injected factory, so a project may
        # return something that does not support binding — losing cache reuse is
        # the right degradation, failing the compaction is not.
        if tools and hasattr(llm, "bind_tools"):
            llm = llm.bind_tools(tools)
    else:
        messages = [SystemMessage(content=instruction), HumanMessage(content=_render(rows))]

    resp = await llm.ainvoke(messages)
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    return content.strip()


async def compact(
    *,
    backend: ConversationBackend,
    llm_factory: Any,
    config: SummarizerConfig,
    conversation_id: Any,
    system_prompt: str | None = None,
    tools: list[Any] | None = None,
    attachments: AttachmentsConfig | None = None,
    force: bool = False,
) -> bool:
    """Compact one conversation. Returns True if a checkpoint was written.

    `force` skips the pressure check and compacts as much as it safely can —
    the recovery path after a provider has already rejected a request as too
    long. Everything else is identical, so the two entry points cannot drift.
    """
    if not config.enabled:
        return False

    cid = str(conversation_id)
    if cid in _in_flight:
        logger.debug("compaction already running conv=%s", cid)
        return False
    _in_flight.add(cid)
    try:
        if not force:
            # Only the pressure check needs the replay set. Recovery already
            # knows the request was too big — the provider said so — and it runs
            # inside the user's turn, so it skips the round-trip.
            replay = await backend.fetch_history(conversation_id, for_llm=True)
            # `attachments` prices the file content hydration will re-attach.
            # Without it a photo reads as its filename and a full conversation
            # looks roomy — the one error direction compaction cannot recover
            # from on its own.
            pressure = estimate_messages(replay, attachments=attachments)
            over_fuse = len(replay) > config.message_threshold
            if pressure < config.threshold_tokens and not over_fuse:
                return False
            logger.info(
                "compaction triggered conv=%s estimated=%d threshold=%d",
                cid, pressure, config.threshold_tokens,
            )

        meta = await backend.fetch_meta(conversation_id)
        already = int(meta.get("summarized_count", 0) or 0)
        full = await backend.fetch_messages_full(conversation_id)

        # Only the span not yet folded is new work. Re-summarizing from message
        # zero every time made each compaction cost more than the last, for a
        # result the previous checkpoint already contained.
        pending = full[already:]
        if force:
            # Recovery keeps the smallest safe tail rather than the configured
            # one — the configured one is what just failed to fit.
            cutoff_in_pending = _align_cutoff(pending, max(0, len(pending) - 2))
        else:
            cutoff_in_pending = _select_cutoff(pending, config, attachments)
        if cutoff_in_pending <= 0:
            logger.info("compaction found nothing safe to fold conv=%s", cid)
            return False

        span = pending[:cutoff_in_pending]
        through_index = already + cutoff_in_pending

        # Cap the new span BEFORE prepending the prior checkpoint. Trimming a
        # combined list from the front could drop the checkpoint itself, which
        # would silently discard everything older than this compaction — the
        # one piece of context that has no other copy in the replay set.
        span, dropped, truncated = _cap_input(span, config)
        if not span:
            return False
        if dropped:
            # `through_index` still covers the dropped rows, so they leave the
            # replay set without being summarized. That is the intended trade:
            # keeping them would mean keeping the very messages that made this
            # span too large to summarize. They are not lost — nothing is
            # deleted, and `recall` can read them back — but the model will not
            # see them again on its own, so say so rather than let it look like
            # a complete checkpoint.
            logger.warning(
                "compaction conv=%s dropped %d rows from the summary input "
                "(span exceeded max_summary_input_ratio)",
                cid, dropped,
            )

        previous = (meta.get("summary") or "").strip()
        to_summarize: list[dict[str, Any]] = []
        if previous:
            # The prior checkpoint leads the span so the model merges into it
            # rather than producing a second, overlapping one. It sits in the
            # same position it occupies in a real request, which keeps the
            # replayed prefix aligned with the cached one.
            to_summarize.append({"role": "user", "content": {"text": previous}})
        to_summarize.extend(span)

        summary = await _summarize(
            llm_factory=llm_factory,
            config=config,
            rows=to_summarize,
            system_prompt=system_prompt,
            tools=tools,
            dropped_prefix=truncated,
        )
        if not summary:
            raise ValueError("summarization returned empty content")

        await backend.save_summary(
            conversation_id, summary=frame_summary(summary), through_index=through_index
        )
        _failures.pop(cid, None)
        logger.info(
            "compacted conv=%s folded=%d through=%d summary_tokens=%d",
            cid, len(span), through_index, estimate_text(summary),
        )
        return True
    except Exception as e:  # noqa: BLE001 — compaction is best-effort by contract
        count = _failures.get(cid, 0) + 1
        _failures[cid] = count
        message = "compaction failed conv=%s attempt=%d: %s: %s"
        args = (cid, count, type(e).__name__, e)
        if count >= 2:
            # Repeated failure means the conversation is growing with nothing
            # holding it back. That is worth an error, not a warning.
            logger.error(message, *args)
        else:
            logger.warning(message, *args)
        return False
    finally:
        _in_flight.discard(cid)


async def maybe_summarize(
    *,
    backend: ConversationBackend,
    llm_factory: Any,
    config: SummarizerConfig,
    conversation_id: Any,
    system_prompt: str | None = None,
    tools: list[Any] | None = None,
    attachments: AttachmentsConfig | None = None,
) -> bool:
    """Compact if the next request would be over threshold. Best-effort."""
    return await compact(
        backend=backend,
        llm_factory=llm_factory,
        config=config,
        conversation_id=conversation_id,
        system_prompt=system_prompt,
        tools=tools,
        attachments=attachments,
        force=False,
    )
