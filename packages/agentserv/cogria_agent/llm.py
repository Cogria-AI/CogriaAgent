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
        return ChatOpenAI(
            model=model or c.model,
            api_key=c.api_key,
            base_url=c.base_url,
            streaming=True,
            stream_usage=True,
            temperature=c.temperature,
        )

    def summary_llm(self) -> ChatOpenAI:
        c = self._c.llm
        return ChatOpenAI(
            model=c.summary_model or c.model,
            api_key=c.api_key,
            base_url=c.base_url,
            streaming=False,
            temperature=c.summary_temperature,
        )

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
