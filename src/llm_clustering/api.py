"""Public API for LLM clustering library."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import pandas as pd
from loguru import logger

from llm_clustering.clustering.registry import ClusterRecord, ClusterRegistry
from llm_clustering.config import Settings, get_settings
from llm_clustering.pipeline.runner import PipelineResult, PipelineRunner

if TYPE_CHECKING:
    from llm_clustering.llm.base import BaseLLMProvider


@dataclass(slots=True)
class ClusteringResult:
    """Result of clustering operation."""

    batch_id: str
    assignments: pd.DataFrame
    clusters: list[ClusterRecord]
    coverage: float
    metrics: dict[str, float | int | str]

    @property
    def total_requests(self) -> int:
        """Total number of requests processed."""
        return len(self.assignments)

    @property
    def assigned_requests(self) -> int:
        """Number of requests assigned to clusters."""
        return int(self.assignments["cluster_id"].notna().sum())


@dataclass(slots=True)
class PartialResult:
    """Result of partial clustering operation."""

    batch_number: int
    batch_id: str
    assignments: pd.DataFrame
    new_clusters: list[ClusterRecord]
    processed_rows: int
    total_rows: int


class ClusteringPipeline:
    """High-level API for LLM-based clustering."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        settings: Settings | None = None,
        business_context: str | None = None,
        registry_path: Path | None = None,
    ) -> None:
        """Initialize clustering pipeline.

        Args:
            llm_provider: Custom LLM provider instance. If None, uses default from settings.
            settings: Configuration settings. If None, loads from environment.
            business_context: Additional business context to add to prompts.
            registry_path: Path to cluster registry file. If None, uses default location.
        """
        self.settings = settings or get_settings()
        self.business_context = business_context
        
        # Initialize registry
        if registry_path:
            self.registry = ClusterRegistry(storage_path=registry_path, settings=self.settings)
        else:
            self.registry = ClusterRegistry(settings=self.settings)
        
        # Store custom LLM provider
        self.llm_provider = llm_provider
        
        logger.info(
            "ClusteringPipeline initialized (llm=%s, business_context=%s)",
            "custom" if llm_provider else "default",
            "provided" if business_context else "none",
        )

    def fit(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        batch_id: str | None = None,
        limit: int | None = None,
    ) -> ClusteringResult:
        """Perform complete clustering on DataFrame.

        Args:
            df: Input DataFrame with text data.
            text_column: Name of column containing text to cluster.
            batch_id: Optional batch identifier. Auto-generated if not provided.
            limit: Optional limit on number of rows to process.

        Returns:
            ClusteringResult with assignments and clusters.
        """
        logger.info(
            "Starting full clustering: rows=%d, text_column=%s, limit=%s",
            len(df),
            text_column,
            limit,
        )

        # Apply limit if specified
        work_df = df.head(limit) if limit else df.copy()

        # Create runner with custom LLM and business context
        runner = self._create_runner(text_column=text_column)
        
        # Run pipeline
        result = runner.run(work_df, batch_id=batch_id)
        
        # Build clustering result
        return self._build_clustering_result(result)

    def fit_partial(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        batch_size: int | None = None,
        start_from: int = 0,
    ) -> Iterator[PartialResult]:
        """Perform iterative clustering on DataFrame in batches.

        Args:
            df: Input DataFrame with text data.
            text_column: Name of column containing text to cluster.
            batch_size: Number of rows per batch. Uses config default if None.
            start_from: Starting row index for processing.

        Yields:
            PartialResult for each processed batch.
        """
        if batch_size is None:
            batch_size = self.settings.clustering_batch_size

        total_rows = len(df)
        logger.info(
            "Starting partial clustering: total_rows=%d, batch_size=%d, start_from=%d",
            total_rows,
            batch_size,
            start_from,
        )

        runner = self._create_runner(text_column=text_column)
        batch_number = 0

        for idx in range(start_from, total_rows, batch_size):
            batch_number += 1
            end_idx = min(idx + batch_size, total_rows)
            batch_df = df.iloc[idx:end_idx].copy()

            logger.info(
                "Processing batch %d: rows %d-%d (%d rows)",
                batch_number,
                idx,
                end_idx,
                len(batch_df),
            )

            # Get clusters before processing
            clusters_before = set(c.cluster_id for c in self.registry.list_clusters())

            # Run pipeline on batch
            result = runner.run(batch_df, batch_id=None)

            # Find new clusters
            clusters_after = {c.cluster_id for c in self.registry.list_clusters()}
            new_cluster_ids = clusters_after - clusters_before
            new_clusters = [
                c for c in self.registry.list_clusters() if c.cluster_id in new_cluster_ids
            ]

            yield PartialResult(
                batch_number=batch_number,
                batch_id=result.batch_id,
                assignments=result.assignments_df,
                new_clusters=new_clusters,
                processed_rows=end_idx,
                total_rows=total_rows,
            )

    def refit(
        self,
        df: pd.DataFrame,
        previous_assignments: pd.DataFrame,
        text_column: str = "text",
    ) -> ClusteringResult:
        """Re-cluster data with existing cluster knowledge.

        This method is useful for refining clusters or processing additional data
        with knowledge from previous clustering runs.

        Args:
            df: Input DataFrame with text data.
            previous_assignments: DataFrame with previous cluster assignments.
            text_column: Name of column containing text to cluster.

        Returns:
            ClusteringResult with updated assignments and clusters.
        """
        logger.info(
            "Starting re-clustering: new_rows=%d, previous_assignments=%d",
            len(df),
            len(previous_assignments),
        )

        # The registry already contains clusters from previous runs
        # We just need to run the pipeline again
        runner = self._create_runner(text_column=text_column)
        result = runner.run(df, batch_id=None)

        return self._build_clustering_result(result)

    def get_clusters(self) -> list[ClusterRecord]:
        """Get all discovered clusters sorted by frequency.

        Returns:
            List of ClusterRecord objects sorted by usage count (descending).
        """
        return self.registry.list_clusters()

    def save_clusters(self, path: Path) -> None:
        """Save cluster registry to a file.

        Args:
            path: Destination path for cluster registry.
        """
        path = Path(path)
        logger.info("Saving clusters to %s", path)
        
        # Copy current registry file to specified path
        shutil.copy2(self.registry.storage_path, path)
        logger.info("Clusters saved successfully")

    def load_clusters(self, path: Path) -> None:
        """Load cluster registry from a file.

        Args:
            path: Source path for cluster registry.
        """
        path = Path(path)
        logger.info("Loading clusters from %s", path)
        
        if not path.exists():
            raise FileNotFoundError(f"Cluster registry not found: {path}")
        
        # Copy specified file to current registry path
        shutil.copy2(path, self.registry.storage_path)
        
        # Reload registry
        self.registry = ClusterRegistry(
            storage_path=self.registry.storage_path,
            settings=self.settings,
        )
        logger.info("Loaded %d clusters", len(self.registry.list_clusters()))

    def _create_runner(self, text_column: str) -> PipelineRunner:
        """Create pipeline runner with current configuration."""
        return PipelineRunner(
            settings=self.settings,
            registry=self.registry,
            text_column=text_column,
            llm_provider=self.llm_provider,
            business_context=self.business_context,
        )

    def _build_clustering_result(self, result: PipelineResult) -> ClusteringResult:
        """Convert PipelineResult to ClusteringResult."""
        total = len(result.assignments_df)
        assigned = int(result.assignments_df["cluster_id"].notna().sum())
        coverage = (assigned / total * 100) if total else 0.0

        return ClusteringResult(
            batch_id=result.batch_id,
            assignments=result.assignments_df,
            clusters=self.registry.list_clusters(),
            coverage=coverage,
            metrics={
                "batch_id": result.batch_id,
                "total_requests": total,
                "assigned_requests": assigned,
                "coverage_pct": round(coverage, 2),
            },
        )

