"""Triton Inference Server LLM provider implementation."""

from typing import Any

import requests
from loguru import logger

from llm_clustering.config import get_settings
from llm_clustering.llm.base import BaseLLMProvider


class TritonProvider(BaseLLMProvider):
    """Triton Inference Server provider for local LLM models."""

    def __init__(self) -> None:
        """Initialize Triton provider."""
        settings = get_settings()
        self.api_url = settings.triton_api_url
        self.model = settings.triton_model or "qwen3_30b_4bit"
        self.temperature = settings.default_temperature
        self.max_tokens = settings.default_max_tokens

    def _make_request(
        self,
        inputs: list[dict[str, Any]],
        outputs: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Make a request to Triton Inference Server API v2.
        
        Args:
            inputs: List of input tensors with name, shape, datatype, and data
            outputs: List of output tensor names (optional, defaults to ["response"])
            
        Returns:
            Response from Triton API
        """
        url = f"{self.api_url}/v2/models/{self.model}/infer"
        headers = {"Content-Type": "application/json"}
        
        if outputs is None:
            outputs = [{"name": "response"}]
        
        payload = {
            "inputs": inputs,
            "outputs": outputs,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Triton API error: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise ConnectionError(
                f"Cannot connect to Triton at {self.api_url}. "
                f"Make sure Triton server is running. Error: {e}"
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.
        
        Note: Triton model may not support embeddings directly.
        This is a placeholder implementation.
        """
        logger.warning("Embeddings via Triton not fully implemented yet")
        # TODO: Implement embeddings if Triton model supports it
        # For now, return zero vectors as fallback
        embeddings = []
        for text in texts:
            # Return zero vector as fallback (768 dimensions)
            embeddings.append([0.0] * 768)
        return embeddings

    def cluster(self, texts: list[str], num_clusters: int | None = None) -> list[int]:
        """Cluster texts using LLM via Triton."""
        # TODO: Implement clustering logic using Triton
        raise NotImplementedError("Clustering logic to be implemented")

    def describe_cluster(self, texts: list[str]) -> str:
        """Generate a description for a cluster using Triton model."""
        prompt = f"""Проанализируй следующие обращения клиентов в контакт-центр и создай краткое описание общей темы или проблемы:

{chr(10).join(f"- {text}" for text in texts[:10])}

Создай краткое описание (1-2 предложения) общей темы этих обращений на русском языке."""

        # Prepare inputs for Triton API v2
        # Note: max_batch_size=0, so no batch dimension needed
        inputs = [
            {
                "name": "text",
                "shape": [1],
                "datatype": "BYTES",
                "data": [prompt],
            },
            {
                "name": "max_tokens",
                "shape": [1],
                "datatype": "INT32",
                "data": [self.max_tokens],
            },
            {
                "name": "temperature",
                "shape": [1],
                "datatype": "FP32",
                "data": [self.temperature],
            },
        ]

        response = self._make_request(inputs)
        
        try:
            # Extract response from Triton output
            outputs = response.get("outputs", [])
            if not outputs:
                raise ValueError("No outputs in Triton response")
            
            # Find response output
            response_output = None
            for output in outputs:
                if output.get("name") == "response":
                    response_output = output
                    break
            
            if response_output is None:
                raise ValueError("Response output not found in Triton response")
            
            # Extract data (can be bytes or string)
            response_data = response_output.get("data", [])
            if not response_data:
                raise ValueError("Empty response data")
            
            description = response_data[0]
            # Decode if bytes
            if isinstance(description, bytes):
                description = description.decode("utf-8")
            elif isinstance(description, str):
                pass  # Already a string
            else:
                description = str(description)
            
            description = description.strip()
            if not description:
                raise ValueError("Empty description from Triton")
            
            return description
        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"Unexpected response format: {response}")
            raise ValueError(f"Failed to extract description from response: {e}") from e

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Make a chat completion request using Triton model.
        
        Note: This method converts messages to a single prompt text,
        as the Triton model expects a single text input.
        """
        # Convert messages to prompt format
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n".join(prompt_parts)

        inputs = [
            {
                "name": "text",
                "shape": [1],
                "datatype": "BYTES",
                "data": [prompt],
            },
            {
                "name": "max_tokens",
                "shape": [1],
                "datatype": "INT32",
                "data": [max_tokens or self.max_tokens],
            },
            {
                "name": "temperature",
                "shape": [1],
                "datatype": "FP32",
                "data": [temperature or self.temperature],
            },
        ]

        response = self._make_request(inputs)
        
        try:
            outputs = response.get("outputs", [])
            if not outputs:
                raise ValueError("No outputs in Triton response")
            
            response_output = None
            for output in outputs:
                if output.get("name") == "response":
                    response_output = output
                    break
            
            if response_output is None:
                raise ValueError("Response output not found")
            
            response_data = response_output.get("data", [])
            if not response_data:
                raise ValueError("Empty response data")
            
            content = response_data[0]
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            elif not isinstance(content, str):
                content = str(content)
            
            return content.strip()
        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"Unexpected response format: {response}")
            raise ValueError(f"Failed to extract content from response: {e}") from e

