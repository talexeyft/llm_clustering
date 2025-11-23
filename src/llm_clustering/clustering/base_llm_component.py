"""Base class for LLM-based components with shared logic."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from llm_clustering.clustering.registry import ClusterRegistry
from llm_clustering.config import Settings, get_settings
from llm_clustering.llm import get_llm_provider
from llm_clustering.llm.prompts import PromptLogEntry, PromptLogger, RenderedPrompt

if TYPE_CHECKING:
    from llm_clustering.llm.base import BaseLLMProvider


class BaseLLMComponent:
    """Base class for Proposer and Judge with shared LLM interaction logic."""

    def __init__(
        self,
        registry: ClusterRegistry | None = None,
        llm: BaseLLMProvider | None = None,
        settings: Settings | None = None,
        business_context: str | None = None,
    ) -> None:
        """Initialize component with LLM and configuration.

        Args:
            registry: Cluster registry for storing/retrieving clusters.
            llm: LLM provider instance. If None, uses default from settings.
            settings: Configuration settings. If None, loads from environment.
            business_context: Optional business context to add to prompts.
        """
        self.settings = settings or get_settings()
        self.registry = registry or ClusterRegistry(settings=self.settings)
        self.llm = llm or get_llm_provider()
        self.prompt_logger = PromptLogger(self.settings)
        self.business_context = business_context

    def _execute_prompt(self, prompt: RenderedPrompt) -> tuple[str, float]:
        """Execute prompt against LLM and return response with latency.

        Args:
            prompt: Rendered prompt with system and user messages.

        Returns:
            Tuple of (response_text, latency_seconds).
        """
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]
        start = time.perf_counter()
        response_text = self.llm.chat_completion(
            messages,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )
        latency = time.perf_counter() - start
        return response_text, latency

    def _parse_json_response(self, response_text: str, error_context: str) -> dict[str, Any]:
        """Parse JSON from LLM response with error handling.

        Args:
            response_text: Raw text response from LLM.
            error_context: Context string for error messages (e.g., "proposer", "judge").

        Returns:
            Parsed JSON as dictionary.

        Raises:
            ValueError: If response is not valid JSON or not a dict.
        """
        from llm_clustering.clustering.utils import extract_json_from_response

        try:
            payload = extract_json_from_response(response_text)
        except json.JSONDecodeError as err:
            logger.error("Failed to parse %s response: %s", error_context, err)
            logger.error("Original response: %s", response_text[:500])
            raise ValueError(f"LLM {error_context} returned invalid JSON.") from err

        if not isinstance(payload, dict):
            raise ValueError(f"LLM {error_context} returned non-object payload.")
        
        return payload

    def _estimate_tokens(self, prompt: RenderedPrompt, response_text: str) -> int:
        """Estimate token count for prompt and response.

        Args:
            prompt: Rendered prompt.
            response_text: Response text from LLM.

        Returns:
            Estimated token count (using simple char/4 heuristic).
        """
        total_chars = len(prompt.system) + len(prompt.user) + len(response_text)
        return max(1, total_chars // 4)

    def _create_log_entry(
        self,
        prompt_name: str,
        batch_id: str,
        slice_id: str,
        prompt: RenderedPrompt,
        response: dict[str, Any],
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> PromptLogEntry:
        """Create a prompt log entry for logging.

        Args:
            prompt_name: Name of the prompt template.
            batch_id: Batch identifier.
            slice_id: Slice identifier.
            prompt: Rendered prompt.
            response: Parsed response dictionary.
            latency_ms: Latency in milliseconds.
            metadata: Optional additional metadata.

        Returns:
            PromptLogEntry ready for logging.
        """
        return PromptLogEntry(
            prompt_name=prompt_name,
            batch_id=batch_id,
            slice_id=slice_id,
            prompt={"system": prompt.system, "user": prompt.user},
            response=response,
            latency_ms=round(latency_ms, 2),
            cost_estimate=None,
            metadata=metadata or {},
        )

