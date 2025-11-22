"""Factory for creating LLM provider instances."""

from loguru import logger

from llm_clustering.config import Settings, get_settings
from llm_clustering.llm.anthropic_provider import AnthropicProvider
from llm_clustering.llm.base import BaseLLMProvider
from llm_clustering.llm.ollama_provider import OllamaProvider
from llm_clustering.llm.openai_provider import OpenAIProvider
from llm_clustering.llm.openrouter_provider import OpenRouterProvider
from llm_clustering.llm.triton_provider import TritonProvider


class LLMFactory:
    """Factory for creating LLM providers."""

    _providers: dict[str, type[BaseLLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "openrouter": OpenRouterProvider,
        "ollama": OllamaProvider,
        "triton": TritonProvider,
    }

    @classmethod
    def create(cls, provider: str | None = None, settings: Settings | None = None) -> BaseLLMProvider:
        """Create an LLM provider instance."""
        settings = settings or get_settings()
        provider_name = cls._resolve_provider(provider, settings)

        if provider_name not in cls._providers:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

        provider_class = cls._providers[provider_name]
        logger.debug("Initializing LLM provider '%s' (model=%s)", provider_name, settings.default_model)
        return provider_class()

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseLLMProvider]) -> None:
        """Register a new LLM provider."""
        cls._providers[name] = provider_class

    @classmethod
    def _resolve_provider(cls, provider: str | None, settings: Settings) -> str:
        """Choose the most suitable provider based on settings and preferences."""
        if provider and provider != "auto":
            return provider

        llm_cfg = settings.llm_config
        requested_provider = llm_cfg.provider

        if requested_provider != "auto":
            return requested_provider

        if llm_cfg.prefer_dual_gpu and "triton" in cls._providers:
            logger.info("LLM config prefers dual GPU backends; selecting 'triton'.")
            return "triton"

        if settings.allow_local_llm_fallback and "ollama" in cls._providers:
            logger.info("Falling back to local Ollama provider as per settings.")
            return "ollama"

        return settings.default_llm_provider


def get_llm_provider(provider: str | None = None) -> BaseLLMProvider:
    """Get an LLM provider instance."""
    return LLMFactory.create(provider)

