"""Default LLMFactory + SystemPromptProvider built from AgentConfig.

The model, base_url, locale names and system prompt all come from config, so the
defaults here suit any domain. A project can pass its own SystemPromptProvider
when it needs a richer prompt (per-user context, retrieved policy, etc.).
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import AgentConfig


class DefaultLLMFactory:
    """Builds ChatOpenAI clients against any OpenAI-compatible endpoint."""

    def __init__(self, config: AgentConfig) -> None:
        self._c = config

    def chat_llm(self, *, model: str | None = None) -> ChatOpenAI:
        c = self._c.llm
        # streaming + stream_usage are required for token-by-token SSE and for
        # persisting input/output token counts from the final chunk. `model`
        # overrides the configured one for a single request (vision turns).
        kwargs: dict = dict(
            model=model or c.model,
            api_key=c.api_key,
            base_url=c.base_url,
            streaming=True,
            stream_usage=True,
        )
        # Omit either knob when unset: the gpt-5 family rejects temperature
        # values != 1, and some gateways reject the rewritten
        # max_completion_tokens outright.
        if c.max_output_tokens is not None:
            kwargs["max_tokens"] = c.max_output_tokens
        if c.temperature is not None:
            kwargs["temperature"] = c.temperature
        # gpt-5.6 on /v1/chat/completions refuses function tools unless
        # reasoning is explicitly off (400 otherwise). Matches the effective
        # gpt-5.2 behaviour; revisit if we move to the Responses API.
        if (kwargs["model"] or "").startswith("gpt-5.6"):
            kwargs["reasoning_effort"] = "none"
        return ChatOpenAI(**kwargs)

    def summary_llm(self) -> ChatOpenAI:
        c = self._c.llm
        kwargs: dict = dict(
            model=c.summary_model or c.model,
            api_key=c.api_key,
            base_url=c.base_url,
            streaming=False,
        )
        if c.summary_temperature is not None:
            kwargs["temperature"] = c.summary_temperature
        return ChatOpenAI(**kwargs)

    def model_name(self) -> str:
        return self._c.llm.model


class ConfigSystemPromptProvider:
    """SystemPromptProvider that renders config.system_prompt with the locale's
    language name. Falls back to the default locale's name for unknown locales."""

    def __init__(self, config: AgentConfig) -> None:
        self._c = config

    def locale_name(self, *, locale: str | None) -> str:
        names = self._c.locales.names
        if locale and locale in names:
            return names[locale]
        return names.get(self._c.locales.default, "English")

    def system_prompt(self, *, locale: str | None) -> str:
        return self._c.system_prompt.format(locale_name=self.locale_name(locale=locale))
