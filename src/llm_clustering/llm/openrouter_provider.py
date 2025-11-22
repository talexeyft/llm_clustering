"""OpenRouter LLM provider implementation for Qwen and other models."""

from typing import Any

import requests
from loguru import logger

from llm_clustering.config import get_settings
from llm_clustering.llm.base import BaseLLMProvider


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider for Qwen 3 Next and other models."""

    API_URL = "https://openrouter.ai/api/v1"

    def __init__(self) -> None:
        """Initialize OpenRouter provider."""
        settings = get_settings()
        self.api_key = settings.openrouter_api_key
        # Qwen 3 Next model identifier
        self.model = settings.openrouter_model or "qwen/qwen-3-next"
        self.temperature = settings.default_temperature
        self.max_tokens = settings.default_max_tokens

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY in .env file"
            )

    def _make_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Make a request to OpenRouter API."""
        url = f"{self.API_URL}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/llm-clustering",  # Optional
            "X-Title": "LLM Clustering",  # Optional
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API error: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenRouter embeddings endpoint."""
        # OpenRouter supports embeddings through some models
        # For now, we'll use a workaround with chat completion
        # TODO: Check if OpenRouter has dedicated embeddings endpoint
        logger.warning("Embeddings via OpenRouter not fully implemented yet")
        # Placeholder - will need to implement based on OpenRouter capabilities
        raise NotImplementedError("Embeddings not yet implemented for OpenRouter")

    def cluster(self, texts: list[str], num_clusters: int | None = None) -> list[int]:
        """Cluster texts using LLM via OpenRouter."""
        # TODO: Implement clustering logic using Qwen 3 Next
        # This will involve creating a prompt that asks the model to cluster texts
        raise NotImplementedError("Clustering logic to be implemented")

    def describe_cluster(self, texts: list[str]) -> str:
        """Generate a description for a cluster using Qwen 3 Next."""
        prompt = f"""Проанализируй следующие обращения клиентов в контакт-центр и создай краткое описание общей темы или проблемы:

{chr(10).join(f"- {text}" for text in texts[:10])}

Создай краткое описание (1-2 предложения) общей темы этих обращений на русском языке."""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = self._make_request("chat/completions", payload)
        
        try:
            description = response["choices"][0]["message"]["content"]
            return description.strip()
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format: {response}")
            raise ValueError(f"Failed to extract description from response: {e}")

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Make a chat completion request."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        response = self._make_request("chat/completions", payload)
        
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format: {response}")
            raise ValueError(f"Failed to extract content from response: {e}")

