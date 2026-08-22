"""AgentConfig — every knob the kernel has, declared in one place.

Locale set, model, base_url, thresholds, app name and JWT issuer are all
configuration rather than constants in the code. A project constructs an
AgentConfig (often from env) and passes it to build_app().
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"  # any OpenAI-compatible gateway
    api_key: str = ""
    model: str = "gpt-4o-mini"
    summary_model: str | None = None  # falls back to `model`
    # None omits the parameter entirely — some model families (gpt-5) reject
    # any explicit value other than their default.
    temperature: float | None = 0.2
    summary_temperature: float | None = 0.3
    # Explicit output budget. Left unset, the effective limit is whatever the
    # gateway defaults to (2048 on some), so provider config silently decides
    # when replies get cut off. Raising it does not prevent a runaway reply —
    # the graph's per-response tool-call guards do that — it stops a
    # legitimately long answer from being truncated at a limit we never chose.
    # None omits the parameter — the escape hatch for gateways that reject it
    # (langchain-openai rewrites it to max_completion_tokens on the wire).
    max_output_tokens: int | None = 4096


class GraphConfig(BaseModel):
    # Hard cap on chat/tool round-trips per /chat request (cost-control).
    # Complex flows rarely exceed 4; 10 leaves headroom without runaway loops.
    max_turns: int = 10
    # Hard cap on how many distinct calls ONE model response may execute.
    # max_turns bounds rounds, not width: a degenerate response carrying dozens
    # of calls runs them all before the round counter is next consulted, and a
    # call can cost real money. Legitimate replies rarely exceed 3-4.
    max_tool_calls_per_turn: int = 8
    # How many times one request may be retried after the provider rejects it
    # for exceeding its context window. Each retry costs a forced compaction
    # plus a fresh request, so one is the useful number: if a maximally
    # compacted history still doesn't fit, another attempt won't change that.
    max_overflow_retries: int = 1


#: The compaction instruction, delivered as the final user message after the
#: replayed conversation. Structured on purpose: a one-line "summarize this"
#: reliably loses the two things this kernel cannot afford to lose — exact
#: identifiers, and the state of a propose/confirm exchange.
DEFAULT_SUMMARY_PROMPT = "\n".join(
    [
        "Condense the conversation ABOVE into a structured checkpoint so the assistant "
        "can continue the work with nothing essential lost.",
        "",
        "Output EXACTLY the sections below, in order. Use terse bullets. Write "
        '"(none)" for an empty section — never drop a section.',
        "",
        "## Intent",
        "- [what the user is trying to accomplish, including how it has shifted]",
        "",
        "## Facts and Identifiers",
        "- [names, numbers, ids, dates, amounts — copied verbatim, never paraphrased]",
        "",
        "## Actions Taken",
        "- [which tools ran, with the arguments and outcomes that still matter]",
        "",
        "## Confirmation State",
        "- [any write action proposed but not yet confirmed, any the user confirmed "
        "or cancelled, and the proposal_token verbatim if one is still open]",
        "",
        "## Open Requests",
        "- [what the user asked for that is not done yet]",
        "",
        "## Current Work",
        "- [precisely what was in progress at this point]",
        "",
        "## Next Step",
        '- [the single next action, or "(none)"]',
        "",
        "Rules:",
        "- Preserve identifiers, numbers, file names, and error strings EXACTLY. "
        "Everything else may be compressed.",
        "- Record the user's corrections and explicit instructions faithfully.",
        "- Do NOT mention this instruction or that the conversation was compacted.",
        "- Output only the checkpoint text. Do not call any tool.",
        "- If a previous checkpoint appears above, merge it: keep what is still "
        "true, drop what is stale, and produce ONE consolidated checkpoint rather "
        "than copying the old one forward.",
    ]
)

#: Framing for the message that replaces the folded span, so the model reads the
#: checkpoint as established background rather than as a fresh instruction.
SUMMARY_PREAMBLE = (
    "This is an automatically generated checkpoint condensing an earlier part of "
    "this conversation to free up context. Treat it as established background and "
    "continue directly from the messages that follow. Do not acknowledge or restate it."
)


class SummarizerConfig(BaseModel):
    """When to compact a conversation, and how much of it to keep verbatim.

    Everything is expressed against `context_window`, because the question being
    answered is "will the next request fit". The previous absolute
    `token_threshold` compared lifetime usage — which grows quadratically, since
    every turn resends the whole history — against a context budget, so it fired
    on conversations that were nowhere near full and stayed quiet on ones that
    were. It is kept only so existing configs still load.
    """

    enabled: bool = True

    # The model's context window. Set this to match `llm.model`; the default is
    # deliberately conservative, because guessing high is the dangerous
    # direction — the threshold ends up above the real window, compaction never
    # fires, and the conversation runs until the provider rejects it.
    context_window: int = 32_000
    # Compact once the next request is estimated to reach this share of it.
    threshold_ratio: float = 0.7
    # How much of the recent tail to keep verbatim, as a share of the window.
    retain_ratio: float = 0.2
    # Floor on the verbatim tail, in messages. `retain_ratio` does the real
    # work; this stops a single oversized message from folding everything.
    keep_recent: int = 4
    # Cap on what one summarization call may read, as a share of the window.
    # Without it a long conversation eventually hands the summary model more
    # than it can take, and the failure is silent: compaction stops working
    # exactly when it is needed most.
    max_summary_input_ratio: float = 0.6
    # Replay the conversation's own system prompt, tools and messages for the
    # summarization call, so it is a genuine prefix of the last request and the
    # provider's prompt cache is reused. Turn off to send a flat transcript
    # instead (correct, just billed at full price).
    reuse_conversation_prefix: bool = True

    # Retained so a config written against the old shape still loads. Unset by
    # default and ignored; `context_window` × `threshold_ratio` decides.
    token_threshold: int | None = None
    # A pure fuse against pathological message counts, not a trigger.
    message_threshold: int = 200

    # Domain-neutral by default. A project may override it to add domain hints
    # (e.g. "keep product names, prices, categories").
    prompt: str = DEFAULT_SUMMARY_PROMPT

    @property
    def threshold_tokens(self) -> int:
        return int(self.context_window * self.threshold_ratio)

    @property
    def retain_tokens(self) -> int:
        return int(self.context_window * self.retain_ratio)

    @property
    def max_summary_input_tokens(self) -> int:
        return int(self.context_window * self.max_summary_input_ratio)


class PruneConfig(BaseModel):
    """Bounds on a single tool result as it enters history.

    A result is written once and resent on every later turn, so an oversized one
    is charged repeatedly. Trimming it costs no model call. Propose/confirm
    envelopes are exempt regardless of size — see prune.py.
    """

    enabled: bool = True
    # Prune a result whose serialized form exceeds this many characters.
    threshold_chars: int = 8_192
    # Retained from each end when a result has to be cut as text.
    head_chars: int = 4_096
    tail_chars: int = 1_024
    # Rows retained when the oversized part is a list — the common case.
    keep_items: int = 20

    def validated(self) -> PruneConfig:
        """Reject a configuration that could not shrink anything."""
        if self.head_chars + self.tail_chars >= self.threshold_chars:
            raise ValueError(
                f"PruneConfig: head_chars + tail_chars ({self.head_chars} + {self.tail_chars}) "
                f"must be less than threshold_chars ({self.threshold_chars}); "
                "otherwise a pruned result would not be smaller than the one that triggered it"
            )
        return self


class RecallConfig(BaseModel):
    """The `history_search` tool: letting the model read back what was folded.

    Off by default, per the kernel's rule that an optional capability costs
    nothing until asked for. Switching it on adds one tool schema to every
    request (and so invalidates a warm prompt prefix once, at deploy time).

    Compaction folds old messages out of the replay set but does NOT delete
    them — `summarized_count` is a cursor, the rows stay in the backend. This
    tool is the path back to them, scoped to the current conversation.
    """

    enabled: bool = False
    # Maximum hits returned to the model in one search.
    max_results: int = 5
    # Characters of surrounding context returned per hit.
    snippet_chars: int = 400


class AttachmentsConfig(BaseModel):
    """Upload limits, extraction budget and the vision policy.

    Attachments are only served when build_app() gets an attachment store +
    repository; this config tunes them, it doesn't switch them on.
    """

    max_file_bytes: int = 20 * 1024 * 1024
    max_files_per_turn: int = 5
    # Per-document and whole-turn caps on injected extracted text. Overflow is
    # truncated with a visible marker rather than silently dropped.
    max_chars_per_doc: int = 20_000
    max_chars_total: int = 40_000
    # Images need a vision-capable model. Unset (the default) means image
    # uploads are refused with a friendly message rather than silently ignored —
    # a text-only model like deepseek-chat cannot see them.
    vision_model: str | None = None
    # How many of the most recent user turns get their images re-sent on replay.
    # Older ones degrade to a one-line placeholder (images are expensive).
    image_history_turns: int = 1
    # Sniffed (magic-byte) mime must be in this allowlist. Deliberately excludes
    # SVG (scriptable) and legacy .doc (no reliable extractor).
    allowed_mimes: list[str] = Field(
        default_factory=lambda: [
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/gif",
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "text/plain",
            "text/markdown",
            "text/csv",
            "application/json",
        ]
    )
    extract_timeout_seconds: float = 30.0
    # Office formats are zip containers: cap the inflated size and entry count
    # so a zip bomb can't take the process down.
    max_uncompressed_bytes: int = 200 * 1024 * 1024
    max_zip_entries: int = 5_000


class AuthConfig(BaseModel):
    # HS256 shared secret with the BFF/business backend that mints the JWT.
    jwt_secret: str = ""
    jwt_issuer: str = "cogria-agent"
    # Claims required on every token. NOTE: no tenant claims (single-tenant).
    required_claims: list[str] = Field(default_factory=lambda: ["exp", "sub"])


class LocaleConfig(BaseModel):
    default: str = "en"
    # locale code -> human language name injected into the system prompt.
    names: dict[str, str] = Field(default_factory=lambda: {"en": "English"})


class AgentConfig(BaseModel):
    app_name: str = "cogria-agentserv"
    env: str = "dev"  # "dev" enables /docs and short cache TTLs
    llm: LLMConfig = Field(default_factory=LLMConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    summarizer: SummarizerConfig = Field(default_factory=SummarizerConfig)
    prune: PruneConfig = Field(default_factory=PruneConfig)
    recall: RecallConfig = Field(default_factory=RecallConfig)
    attachments: AttachmentsConfig = Field(default_factory=AttachmentsConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    locales: LocaleConfig = Field(default_factory=LocaleConfig)
    # Inline system prompt template. `{locale_name}` is substituted with the
    # reply language name. A project may instead supply a SystemPromptProvider.
    system_prompt: str = (
        "You are a helpful assistant. Reply in {locale_name} throughout, even if "
        "the user writes in another language.\n\n"
        "- When the user asks for data, call the most relevant tool, then explain "
        "the result in natural language — do not dump raw JSON.\n"
        "- For write actions you will be asked to confirm first (propose/confirm); "
        "do not restate the proposal, just wait for the user.\n"
        "- When you see a user message of `[CONFIRMED]`, immediately re-call the "
        "same write tool with the supplied proposal_token. When you see "
        "`[CANCELLED]`, briefly acknowledge and call nothing.\n"
        "- If you don't know something, say so. Never fabricate data."
    )

    @classmethod
    def from_env(cls) -> AgentConfig:
        """Build from environment variables (the common deployment path)."""
        return cls(
            app_name=os.environ.get("AGENT_APP_NAME", "cogria-agentserv"),
            env=os.environ.get("AGENTSERV_ENV", "dev"),
            llm=LLMConfig(
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                model=os.environ.get("AGENT_MODEL", "gpt-4o-mini"),
                summary_model=os.environ.get("AGENT_SUMMARY_MODEL") or None,
            ),
            summarizer=SummarizerConfig(
                # Sized to the model in use; the conservative default only fits
                # the smallest windows. See SummarizerConfig.context_window.
                context_window=int(os.environ.get("AGENT_CONTEXT_WINDOW") or 32_000),
            ),
            recall=RecallConfig(
                enabled=(os.environ.get("AGENT_RECALL", "").lower() in {"1", "true", "yes"}),
            ),
            auth=AuthConfig(
                jwt_secret=os.environ.get("JWT_SECRET", ""),
                jwt_issuer=os.environ.get("JWT_ISSUER", "cogria-agent"),
            ),
            attachments=AttachmentsConfig(
                # Unset means image uploads are refused (a text-only model can't
                # read them); documents work either way.
                vision_model=os.environ.get("AGENT_VISION_MODEL") or None,
            ),
        )
