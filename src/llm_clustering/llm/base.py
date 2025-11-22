"""Base class for LLM providers."""

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def cluster(self, texts: list[str], num_clusters: int | None = None) -> list[int]:
        """Cluster texts using LLM."""
        pass

    @abstractmethod
    def describe_cluster(self, texts: list[str]) -> str:
        """Generate a description for a cluster of texts."""
        pass

    @abstractmethod
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Run a chat-style completion request."""
        pass

