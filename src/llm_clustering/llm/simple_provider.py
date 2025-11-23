"""Simplified base class for custom LLM providers.

For most use cases, you only need to implement chat_completion() method.
Other methods have default implementations or can be left as NotImplementedError.
"""

from abc import ABC, abstractmethod

from llm_clustering.llm.base import BaseLLMProvider


class SimpleLLMProvider(BaseLLMProvider, ABC):
    """Simplified LLM provider - only chat_completion() is required.
    
    This is a convenience base class for users who want to add their own LLM
    without implementing all the abstract methods. Most clustering functionality
    only needs chat_completion() and describe_cluster() methods.
    
    Example:
        ```python
        class MyLLM(SimpleLLMProvider):
            def __init__(self, api_key: str):
                self.api_key = api_key
                self.api_url = "https://my-llm-api.com"
            
            def chat_completion(self, messages, temperature=None, max_tokens=None):
                # Your LLM API call here
                import requests
                response = requests.post(
                    f"{self.api_url}/chat",
                    json={"messages": messages, "temperature": temperature},
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                return response.json()["content"]
        
        # Use it
        pipeline = ClusteringPipeline(llm_provider=MyLLM(api_key="xxx"))
        ```
    """
    
    @abstractmethod
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Run a chat-style completion request.
        
        This is the ONLY method you must implement.
        
        Args:
            messages: List of chat messages in OpenAI format:
                [{"role": "user", "content": "Your prompt"}]
            temperature: Optional temperature for generation (0.0-1.0)
            max_tokens: Optional max tokens to generate
        
        Returns:
            Generated text response from the LLM
        """
        pass
    
    def describe_cluster(self, texts: list[str]) -> str:
        """Generate a description for a cluster of texts.
        
        Default implementation uses chat_completion() with a simple prompt.
        You can override this if you need custom behavior.
        
        Args:
            texts: List of texts in the cluster
        
        Returns:
            Brief description of the cluster theme (1-2 sentences)
        """
        # Limit to first 10 texts to avoid token limits
        sample_texts = texts[:10]
        
        prompt = f"""Проанализируй следующие обращения клиентов и создай краткое описание общей темы или проблемы:

{chr(10).join(f"- {text}" for text in sample_texts)}

Создай краткое описание (1-2 предложения) общей темы этих обращений на русском языке."""
        
        return self.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts.
        
        Default implementation raises NotImplementedError.
        Most clustering pipelines don't need embeddings.
        
        Override this only if your pipeline specifically requires embeddings.
        """
        raise NotImplementedError(
            "Embeddings not implemented. "
            "This method is optional for most use cases."
        )
    
    def cluster(self, texts: list[str], num_clusters: int | None = None) -> list[int]:
        """Cluster texts using LLM.
        
        Default implementation raises NotImplementedError.
        Most clustering pipelines don't use this method directly.
        
        Override this only if you have a specific clustering algorithm.
        """
        raise NotImplementedError(
            "Clustering not implemented. "
            "This method is optional for most use cases."
        )


