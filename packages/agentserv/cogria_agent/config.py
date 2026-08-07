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


class SummarizerConfig(BaseModel):
    enabled: bool = True
    token_threshold: int = 16_000
    message_threshold: int = 50
    keep_recent: int = 20
    # Generic, domain-neutral summary instruction. A project may override it to
    # add domain hints (e.g. "keep product names, prices, categories").
    prompt: str = (
        "Summarize the following assistant-user conversation history into a concise "
        "running memory (max 200 words). Preserve concrete facts the assistant may "
        "need later: names, numbers, identifiers, decisions made, and any pending actions."
    )


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
