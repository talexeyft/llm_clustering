"""Batch pipeline runner that orchestrates builder → proposer → judge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import pandas as pd
from tqdm import tqdm

from llm_clustering.clustering.registry import ClusterRecord, ClusterRegistry
from llm_clustering.config import Settings, get_settings
from llm_clustering.monitoring import CohesionChecker
from llm_clustering.pipeline.batch_builder import (
    BatchBuilder,
    BatchBuildResult,
    BatchSlice,
    SnapshotPaths,
)
from llm_clustering.utils.metrics import MetricsTracker

if TYPE_CHECKING:
    from llm_clustering.clustering.judge import AssignmentJudge, AssignmentResult
    from llm_clustering.clustering.proposer import ClusterProposer, ProposerResult


@dataclass(slots=True)
class SliceOutcome:
    """Result of running proposer + judge for a single slice."""

    slice: BatchSlice
    proposer: ProposerResult
    assignments: list[AssignmentResult]

    @property
    def coverage(self) -> float:
        """Return share of requests with a cluster id."""
        if not self.assignments:
            return 0.0
        assigned = sum(1 for item in self.assignments if item.cluster_id)
        return assigned / len(self.assignments)


@dataclass(slots=True)
class PipelineResult:
    """Full outcome of a pipeline run."""

    batch_id: str
    prepared_df: pd.DataFrame
    prepared_snapshot: SnapshotPaths
    slices: list[SliceOutcome]
    assignments_path: Path
    assignments_df: pd.DataFrame
    metrics_path: Path
    cohesion_report: Path | None


class PipelineRunner:
    """High-level orchestrator that runs the MVP flow end-to-end."""

    def __init__(
        self,
        settings: Settings | None = None,
        registry: ClusterRegistry | None = None,
        text_column: str = "text",
        llm_provider: BaseLLMProvider | None = None,
        business_context: str | None = None,
    ) -> None:
        # Late import to avoid circular dependency
        from llm_clustering.clustering.judge import AssignmentJudge
        from llm_clustering.clustering.proposer import ClusterProposer
        from llm_clustering.llm.base import BaseLLMProvider
        
        self.settings = settings or get_settings()
        self.registry = registry or ClusterRegistry(settings=self.settings)
        self.builder = BatchBuilder(settings=self.settings, text_column=text_column)
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

    def run(self, dataframe: pd.DataFrame, batch_id: str | None = None) -> PipelineResult:
        """Execute the full pipeline on the provided dataframe."""
        build_result = self.builder.build(dataframe, batch_id=batch_id)
        slice_outcomes: list[SliceOutcome] = []
        assignment_rows: list[AssignmentResult] = []

        total_slices = len(build_result.slices)
        processed_requests = 0
        total_requests = len(dataframe)
        
        with tqdm(total=total_slices, desc="Processing batches", unit="batch") as pbar:
            for batch_slice in build_result.slices:
                proposer_result = self.proposer.propose(batch_slice)
                candidate_clusters = self._candidate_clusters(proposer_result.clusters)
                assignments = self.judge.judge_slice(
                    batch_slice,
                    candidate_clusters=candidate_clusters,
                )

                slice_outcomes.append(
                    SliceOutcome(
                        slice=batch_slice,
                        proposer=proposer_result,
                        assignments=assignments,
                    )
                )
                assignment_rows.extend(assignments)
                
                processed_requests += batch_slice.size
                pbar.set_postfix({
                    'requests': f'{processed_requests}/{total_requests}',
                    'coverage': f'{len([a for a in assignment_rows if a.cluster_id])}/{processed_requests}'
                })
                pbar.update(1)

        assignments_df = self._to_dataframe(assignment_rows)
        assignments_path = self._persist_assignments(
            batch_id=build_result.batch_id,
            dataframe=assignments_df,
        )

        metrics_row = self._build_metrics(
            batch_id=build_result.batch_id,
            assignments_df=assignments_df,
            slices=slice_outcomes,
        )
        metrics_path = self.metrics_tracker.append(metrics_row)
        cohesion_report = self.cohesion_checker.run(
            batch_id=build_result.batch_id,
            assignments=assignments_df,
            prepared_df=build_result.prepared,
        )

        return PipelineResult(
            batch_id=build_result.batch_id,
            prepared_df=build_result.prepared,
            prepared_snapshot=build_result.prepared_snapshot,
            slices=slice_outcomes,
            assignments_path=assignments_path,
            assignments_df=assignments_df,
            metrics_path=metrics_path,
            cohesion_report=cohesion_report,
        )

    def _candidate_clusters(
        self,
        new_clusters: Sequence[ClusterRecord],
        limit: int = 25,
    ) -> list[ClusterRecord]:
        existing = self.registry.list_clusters(limit=limit)
        merged: dict[str, ClusterRecord] = {record.cluster_id: record for record in existing}
        for record in new_clusters:
            merged[record.cluster_id] = record
        return list(merged.values())

    def _to_dataframe(self, assignments: Sequence[AssignmentResult]) -> pd.DataFrame:
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
        parquet_path = self.results_dir / f"{batch_id}.parquet"
        csv_path = self.results_dir / f"{batch_id}.csv"
        dataframe.to_csv(csv_path, index=False)
        try:
            dataframe.to_parquet(parquet_path, index=False)
            return parquet_path
        except (ImportError, ValueError):
            # Fall back to CSV path when parquet is unavailable
            return csv_path

    def _build_metrics(
        self,
        batch_id: str,
        assignments_df: pd.DataFrame,
        slices: Sequence[SliceOutcome],
    ) -> dict[str, float | int | str]:
        total = len(assignments_df.index)
        assigned = int(assignments_df["cluster_id"].notna().sum()) if total else 0
        skipped = int((assignments_df["decision"] == "skip").sum()) if total else 0
        coverage_pct = (assigned / total * 100) if total else 0.0
        skipped_pct = (skipped / total * 100) if total else 0.0
        proposer_latency = sum(slice_result.proposer.latency_ms for slice_result in slices)
        judge_latency = sum(
            assignment.latency_ms for slice_result in slices for assignment in slice_result.assignments
        )
        token_estimate = sum(slice_result.proposer.token_estimate for slice_result in slices) + sum(
            assignment.token_estimate for slice_result in slices for assignment in slice_result.assignments
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

