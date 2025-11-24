"""Ollama LLM provider implementation for local models."""

from typing import Any
import json

import urllib3
from loguru import logger

from llm_clustering.config import get_settings
from llm_clustering.llm.base import BaseLLMProvider

# Отключаем предупреждения urllib3
urllib3.disable_warnings()


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for local models like Qwen3 30B."""

    def __init__(self) -> None:
        """Initialize Ollama provider."""
        settings = get_settings()
        self.api_url = settings.ollama_api_url
        self.model = settings.ollama_model or "dengcao/Qwen3-30B-A3B-Instruct-2507"
        self.temperature = settings.default_temperature
        self.max_tokens = settings.default_max_tokens

    def _make_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Make a request to Ollama API.
        
        Uses urllib3 directly instead of requests due to compatibility issues
        with Ollama server (requests returns 503, urllib3 works correctly).
        """
        url = f"{self.api_url}/{endpoint}"
        
        # Парсим URL для urllib3
        if url.startswith("http://"):
            url = url[7:]  # Убираем http://
        elif url.startswith("https://"):
            url = url[8:]  # Убираем https://
        
        host_port = url.split("/", 1)
        host = host_port[0]
        path = "/" + host_port[1] if len(host_port) > 1 else "/"
        
        if ":" in host:
            host, port = host.split(":")
            port = int(port)
        else:
            port = 11434 if "localhost" in host or "127.0.0.1" in host else 80
        
        try:
            # Используем urllib3 напрямую
            http = urllib3.PoolManager(
                num_pools=1,
                maxsize=1,
                timeout=urllib3.Timeout(connect=10, read=120)
            )
            
            response = http.request(
                "POST",
                f"http://{host}:{port}{path}",
                body=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Connection": "close"
                }
            )
            
            if response.status != 200:
                error_msg = f"Ollama API error: {response.status}"
                logger.error(error_msg)
                if response.data:
                    logger.error(f"Response: {response.data.decode('utf-8', errors='ignore')[:200]}")
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.api_url}. "
                    f"Status: {response.status}. "
                    "Make sure Ollama is running: 'ollama serve'"
                )
            
            return json.loads(response.data.decode("utf-8"))
            
        except urllib3.exceptions.HTTPError as e:
            logger.error(
                f"Ollama connection error. Is Ollama running on {self.api_url}? Error: {e}"
            )
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.api_url}. "
                "Make sure Ollama is running: 'ollama serve'"
            ) from e
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse Ollama response: {e}")
            raise ValueError(f"Invalid response from Ollama: {e}") from e
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Ollama embeddings endpoint."""
        # Ollama supports embeddings through /api/embeddings
        embeddings = []
        for text in texts:
            payload = {
                "model": self.model,
                "prompt": text,
            }
            try:
                response = self._make_request("embeddings", payload)
                embeddings.append(response.get("embedding", []))
            except Exception as e:
                logger.warning(f"Failed to generate embedding for text: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * 768)  # Default embedding size
        return embeddings

    def cluster(self, texts: list[str], num_clusters: int | None = None) -> list[int]:
        """Cluster texts using LLM via Ollama."""
        # TODO: Implement clustering logic using Ollama
        raise NotImplementedError("Clustering logic to be implemented")

    def describe_cluster(self, texts: list[str]) -> str:
        """Generate a description for a cluster using Ollama."""
        prompt = f"""Проанализируй следующие обращения клиентов в контакт-центр и создай краткое описание общей темы или проблемы:

{chr(10).join(f"- {text}" for text in texts[:10])}

Создай краткое описание (1-2 предложения) общей темы этих обращений на русском языке."""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        response = self._make_request("generate", payload)
        
        try:
            description = response.get("response", "").strip()
            if not description:
                raise ValueError("Empty response from Ollama")
            return description
        except (KeyError, ValueError) as e:
            logger.error(f"Unexpected response format: {response}")
            raise ValueError(f"Failed to extract description from response: {e}")

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Make a chat completion request using Ollama chat API."""
        # Ollama supports /api/chat endpoint with messages format
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }

        response = self._make_request("chat", payload)
        
        try:
            return response.get("message", {}).get("content", "").strip()
        except (KeyError, AttributeError) as e:
            logger.error(f"Unexpected response format: {response}")
            raise ValueError(f"Failed to extract content from response: {e}")

