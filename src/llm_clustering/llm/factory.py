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
        
        # Explicit provider requested
        if provider and provider != "auto":
            logger.info("Using explicitly requested LLM provider: %s", provider)
            return cls._create_provider(provider, settings)
        
        # Resolve provider from settings
        provider_name = cls._resolve_provider(provider, settings)
        logger.info("Selected LLM provider: %s", provider_name)
        return cls._create_provider(provider_name, settings)
    
    @classmethod
    def _create_provider(cls, provider_name: str, settings: Settings) -> BaseLLMProvider:
        """Create provider instance with error handling."""
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown LLM provider: {provider_name}. Available: {available}"
            )

        provider_class = cls._providers[provider_name]
        logger.debug("Initializing %s with model=%s", provider_name, settings.default_model)
        return provider_class()

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseLLMProvider]) -> None:
        """Register a new LLM provider."""
        cls._providers[name] = provider_class

    @classmethod
    def _resolve_provider(cls, provider: str | None, settings: Settings) -> str:
        """Choose the most suitable provider based on settings and preferences."""
        # Already resolved provider from settings
        if settings.llm_provider != "auto":
            return settings.llm_provider
        
        # Auto-selection logic with detailed logging
        if settings.llm_prefer_dual_gpu and "triton" in cls._providers:
            logger.info("Auto-select: Using 'triton' provider (prefer_dual_gpu=True)")
            return "triton"
        
        if settings.llm_allow_local_fallback and "ollama" in cls._providers:
            logger.info("Auto-select: Using 'ollama' provider (allow_local_llm_fallback=True)")
            return "ollama"
        
        fallback = settings.default_llm_provider
        logger.info("Auto-select: Using default fallback provider '%s'", fallback)
        return fallback


def get_llm_provider(provider: str | None = None) -> BaseLLMProvider:
    """Get an LLM provider instance."""
    return LLMFactory.create(provider)

