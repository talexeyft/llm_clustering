"""OpenAI LLM provider implementation."""

from llm_clustering.config import get_settings
from llm_clustering.llm.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider for embeddings and clustering."""

    def __init__(self) -> None:
        """Initialize OpenAI provider."""
        settings = get_settings()
        self.api_key = settings.openai_api_key
        self.model = settings.default_model
        self.temperature = settings.default_temperature
        self.max_tokens = settings.default_max_tokens

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI."""
        # TODO: Implement OpenAI embeddings
        raise NotImplementedError

    def cluster(self, texts: list[str], num_clusters: int | None = None) -> list[int]:
        """Cluster texts using OpenAI."""
        # TODO: Implement OpenAI-based clustering
        raise NotImplementedError

    def describe_cluster(self, texts: list[str]) -> str:
        """Generate a description for a cluster using OpenAI."""
        # TODO: Implement cluster description
        raise NotImplementedError

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Chat completion placeholder for OpenAI provider."""
        raise NotImplementedError("Chat completion has not been implemented for OpenAIProvider yet.")

