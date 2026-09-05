"""Pluggable LLM provider abstraction.

The rest of the system (search, semantic retrieval, document
management) has zero dependency on this module. QAService uses it only
to optionally generate a synthesized answer on top of retrieval
results; if no provider is configured, `NoOpLLMProvider` is used and
the QA endpoint returns retrieved passages without a generated answer,
clearly marked as such.

Adding a real provider means implementing `LLMProvider.generate` and
registering it in `get_llm_provider()`. `OpenAIProvider` is included as
an example of a paid-API-backed implementation; it is never used
unless the operator explicitly sets LLM_PROVIDER=openai and supplies
LLM_API_KEY, keeping the core application free of any hard dependency
on a paid service.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a completion for the given prompt. Raises on failure."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider is configured and ready to use."""


class NoOpLLMProvider(LLMProvider):
    """Used when LLM_PROVIDER=none (the default). Retrieval-only mode."""

    def generate(self, prompt: str) -> str:
        raise RuntimeError("No LLM provider is configured; generation unavailable.")

    def is_available(self) -> bool:
        return False


class OpenAIProvider(LLMProvider):
    """Optional paid-API provider. Only instantiated if LLM_PROVIDER=openai
    and LLM_API_KEY is set. Not required to run the core application."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str) -> str:
        if not self.is_available():
            raise RuntimeError("OpenAI provider is not configured (missing API key).")
        try:
            from openai import OpenAI  # imported lazily; optional dependency
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is not installed. Install it or set "
                "LLM_PROVIDER=none to use retrieval-only mode."
            ) from exc

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""


_provider_instance: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    if settings.LLM_PROVIDER == "openai" and settings.LLM_API_KEY:
        logger.info("Using OpenAIProvider for RAG answer generation.")
        _provider_instance = OpenAIProvider(settings.LLM_API_KEY, settings.LLM_MODEL)
    else:
        logger.info(
            "No LLM provider configured (LLM_PROVIDER=%s); QA will run in "
            "retrieval-only mode.",
            settings.LLM_PROVIDER,
        )
        _provider_instance = NoOpLLMProvider()

    return _provider_instance
