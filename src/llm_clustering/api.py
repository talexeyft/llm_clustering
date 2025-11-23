"""Public API for LLM clustering library."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Sequence

import pandas as pd
from loguru import logger

from llm_clustering.clustering.judge import AssignmentJudge, AssignmentResult
from llm_clustering.clustering.proposer import ClusterProposer, ProposerResult
from llm_clustering.clustering.registry import ClusterRecord, ClusterRegistry
from llm_clustering.config import Settings, get_settings
from llm_clustering.monitoring import CohesionChecker
from llm_clustering.pipeline.batch_builder import BatchBuilder, BatchSlice
from llm_clustering.utils.metrics import MetricsTracker

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
        self.llm_provider = llm_provider
        
        # Initialize registry
        if registry_path:
            self.registry = ClusterRegistry(storage_path=registry_path, settings=self.settings)
        else:
            self.registry = ClusterRegistry(settings=self.settings)
        
        # Initialize components directly (no PipelineRunner)
        self.builder = BatchBuilder(settings=self.settings, text_column="text")
        self.proposer = ClusterProposer(
            registry=self.registry,
            settings=self.settings,
            llm=llm_provider,
            business_context=business_context,
        )
        self.judge = AssignmentJudge(
            registry=self.registry,
            settings=self.settings,
            llm=llm_provider,
            business_context=business_context,
        )
        self.metrics_tracker = MetricsTracker(settings=self.settings)
        self.cohesion_checker = CohesionChecker(settings=self.settings)
        self.results_dir = Path(self.settings.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
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

        # Update builder's text column
        self.builder.text_column = text_column
        
        # Execute pipeline
        build_result = self.builder.build(work_df, batch_id=batch_id)
        slice_outcomes = []
        assignment_rows = []
        
        total_slices = len(build_result.slices)
        processed_requests = 0
        total_requests = len(work_df)
        
        from tqdm import tqdm
        with tqdm(total=total_slices, desc="Processing batches", unit="batch") as pbar:
            for batch_slice in build_result.slices:
                proposer_result = self.proposer.propose(batch_slice)
                candidate_clusters = self._candidate_clusters(proposer_result.clusters)
                assignments = self.judge.judge_slice(
                    batch_slice,
                    candidate_clusters=candidate_clusters,
                )
                
                slice_outcomes.append({
                    "slice": batch_slice,
                    "proposer": proposer_result,
                    "assignments": assignments,
                })
                assignment_rows.extend(assignments)
                
                processed_requests += batch_slice.size
                pbar.set_postfix({
                    'requests': f'{processed_requests}/{total_requests}',
                    'coverage': f'{len([a for a in assignment_rows if a.cluster_id])}/{processed_requests}'
                })
                pbar.update(1)
        
        # Convert assignments to DataFrame
        assignments_df = self._to_dataframe(assignment_rows)
        assignments_path = self._persist_assignments(
            batch_id=build_result.batch_id,
            dataframe=assignments_df,
        )
        
        # Build and persist metrics
        metrics_row = self._build_metrics(
            batch_id=build_result.batch_id,
            assignments_df=assignments_df,
            slices=slice_outcomes,
        )
        metrics_path = self.metrics_tracker.append(metrics_row)
        
        # Run cohesion check
        cohesion_report = self.cohesion_checker.run(
            batch_id=build_result.batch_id,
            assignments=assignments_df,
            prepared_df=build_result.prepared,
        )
        
        # Build clustering result
        return self._build_clustering_result_from_data(
            batch_id=build_result.batch_id,
            assignments_df=assignments_df,
        )

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
            batch_size = self.settings.batch_size

        total_rows = len(df)
        logger.info(
            "Starting partial clustering: total_rows=%d, batch_size=%d, start_from=%d",
            total_rows,
            batch_size,
            start_from,
        )

        # Update builder's text column
        self.builder.text_column = text_column
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

            # Run pipeline on batch (inline without PipelineRunner)
            build_result = self.builder.build(batch_df, batch_id=None)
            assignment_rows = []
            
            for batch_slice in build_result.slices:
                proposer_result = self.proposer.propose(batch_slice)
                candidate_clusters = self._candidate_clusters(proposer_result.clusters)
                assignments = self.judge.judge_slice(batch_slice, candidate_clusters)
                assignment_rows.extend(assignments)
            
            assignments_df = self._to_dataframe(assignment_rows)

            # Find new clusters
            clusters_after = {c.cluster_id for c in self.registry.list_clusters()}
            new_cluster_ids = clusters_after - clusters_before
            new_clusters = [
                c for c in self.registry.list_clusters() if c.cluster_id in new_cluster_ids
            ]

            yield PartialResult(
                batch_number=batch_number,
                batch_id=build_result.batch_id,
                assignments=assignments_df,
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
        # Just run fit() which now contains the full logic
        return self.fit(df, text_column=text_column, batch_id=None, limit=None)

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

    def _candidate_clusters(
        self,
        new_clusters: Sequence[ClusterRecord],
        limit: int = 25,
    ) -> list[ClusterRecord]:
        """Merge new clusters with existing top clusters."""
        existing = self.registry.list_clusters(limit=limit)
        merged: dict[str, ClusterRecord] = {record.cluster_id: record for record in existing}
        for record in new_clusters:
            merged[record.cluster_id] = record
        return list(merged.values())
    
    def _to_dataframe(self, assignments: Sequence[AssignmentResult]) -> pd.DataFrame:
        """Convert list of AssignmentResult to DataFrame."""
        records = []
        for item in assignments:
            records.append(
                {
                    "batch_id": item.batch_id,
                    "slice_id": item.slice_id,
                    "request_id": item.request_id,
                    "cluster_id": item.cluster_id,
                    "decision": item.decision,
                    "confidence_text": item.confidence_text,
                    "llm_rationale": item.llm_rationale,
                    "suggested_cluster_id": item.suggested_cluster.cluster_id
                    if item.suggested_cluster
                    else None,
                }
            )
        return pd.DataFrame.from_records(records)
    
    def _persist_assignments(self, batch_id: str, dataframe: pd.DataFrame) -> Path:
        """Persist assignments to disk."""
        if not self.settings.save_results:
            # Return a dummy path if saving is disabled
            return self.results_dir / f"{batch_id}.csv"
        
        parquet_path = self.results_dir / f"{batch_id}.parquet"
        csv_path = self.results_dir / f"{batch_id}.csv"
        dataframe.to_csv(csv_path, index=False)
        try:
            dataframe.to_parquet(parquet_path, index=False)
            return parquet_path
        except (ImportError, ValueError):
            return csv_path
    
    def _build_metrics(
        self,
        batch_id: str,
        assignments_df: pd.DataFrame,
        slices: Sequence[dict],
    ) -> dict[str, float | int | str]:
        """Build metrics dictionary from results."""
        total = len(assignments_df.index)
        assigned = int(assignments_df["cluster_id"].notna().sum()) if total else 0
        skipped = int((assignments_df["decision"] == "skip").sum()) if total else 0
        coverage_pct = (assigned / total * 100) if total else 0.0
        skipped_pct = (skipped / total * 100) if total else 0.0
        proposer_latency = sum(s["proposer"].latency_ms for s in slices)
        judge_latency = sum(
            a.latency_ms for s in slices for a in s["assignments"]
        )
        token_estimate = sum(s["proposer"].token_estimate for s in slices) + sum(
            a.token_estimate for s in slices for a in s["assignments"]
        )

        return {
            "batch_id": batch_id,
            "requests_total": total,
            "requests_assigned": assigned,
            "coverage_pct": round(coverage_pct, 2),
            "skipped_pct": round(skipped_pct, 2),
            "proposer_latency_ms": round(proposer_latency, 2),
            "judge_latency_ms": round(judge_latency, 2),
            "token_estimate": token_estimate,
        }

    def _build_clustering_result_from_data(
        self,
        batch_id: str,
        assignments_df: pd.DataFrame,
    ) -> ClusteringResult:
        """Build ClusteringResult from assignments DataFrame."""
        total = len(assignments_df)
        assigned = int(assignments_df["cluster_id"].notna().sum())
        coverage = (assigned / total * 100) if total else 0.0

        return ClusteringResult(
            batch_id=batch_id,
            assignments=assignments_df,
            clusters=self.registry.list_clusters(),
            coverage=coverage,
            metrics={
                "batch_id": batch_id,
                "total_requests": total,
                "assigned_requests": assigned,
                "coverage_pct": round(coverage, 2),
            },
        )

