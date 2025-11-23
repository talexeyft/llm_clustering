"""LLM-based clustering for contact center customer inquiries.

This library provides high-level API for clustering text data using LLM models.

Example:
    >>> from llm_clustering import ClusteringPipeline
    >>> import pandas as pd
    >>> 
    >>> # Create pipeline with default settings
    >>> pipeline = ClusteringPipeline()
    >>> 
    >>> # Cluster your data
    >>> df = pd.DataFrame({"text": ["request 1", "request 2", ...]})
    >>> result = pipeline.fit(df, text_column="text")
    >>> 
    >>> # Get results
    >>> print(f"Coverage: {result.coverage:.1f}%")
    >>> print(f"Clusters: {len(result.clusters)}")
"""

__version__ = "0.1.0"

# Public API exports
from llm_clustering.api import ClusteringPipeline, ClusteringResult, PartialResult
from llm_clustering.clustering.registry import ClusterRecord, ClusterRegistry
from llm_clustering.config import Settings
from llm_clustering.llm.base import BaseLLMProvider
from llm_clustering.llm.simple_provider import SimpleLLMProvider

__all__ = [
    # Main API
    "ClusteringPipeline",
    "ClusteringResult",
    "PartialResult",
    # Cluster management
    "ClusterRecord",
    "ClusterRegistry",
    # Configuration
    "Settings",
    # LLM integration
    "BaseLLMProvider",
    "SimpleLLMProvider",
    # Version
    "__version__",
]
