"""LLM Cluster Proposer implementation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd
from loguru import logger

from llm_clustering.clustering.base_llm_component import BaseLLMComponent
from llm_clustering.clustering.registry import ClusterRecord, ClusterRegistry
from llm_clustering.clustering.schemas import ClusterProposal, ProposerResponse
from llm_clustering.config import Settings
from llm_clustering.llm.base import BaseLLMProvider
from llm_clustering.llm.prompts import (
    RenderedPrompt,
    render_cluster_proposer_prompt,
)
from llm_clustering.pipeline.batch_builder import BatchSlice


@dataclass(slots=True)
class ProposerResult:
    """Structured response from proposer stage."""

    batch_id: str
    slice_id: str
    clusters: list[ClusterRecord]
    skipped_request_ids: list[str]
    raw_response: dict[str, Any]
    latency_ms: float
    token_estimate: int

    @property
    def num_clusters(self) -> int:
        return len(self.clusters)


class ClusterProposer(BaseLLMComponent):
    """Packages slice data, sends prompt to the LLM, and persists clusters."""

    def __init__(
        self,
        registry: ClusterRegistry | None = None,
        llm: BaseLLMProvider | None = None,
        settings: Settings | None = None,
        business_context: str | None = None,
    ) -> None:
        super().__init__(registry, llm, settings, business_context)

    def propose(
        self,
        batch_slice: BatchSlice,
        known_clusters_limit: int = 12,
    ) -> ProposerResult:
        """Generate and persist clusters for a batch slice."""
        prompt = self._build_prompt(batch_slice, known_clusters_limit)
        response_text, latency = self._execute_prompt(prompt)
        latency_ms = latency * 1000
        token_estimate = self._estimate_tokens(prompt, response_text)
        
        # Parse and validate response using Pydantic
        json_data = self._parse_json_response(response_text, "proposer")
        try:
            response = ProposerResponse.model_validate(json_data)
        except Exception as err:
            logger.error("Failed to validate proposer response: %s", err)
            # Fallback to empty response
            response = ProposerResponse(batch_id=batch_slice.batch_id, clusters=[], skipped_request_ids=[])

        # Persist clusters from validated response
        clusters = self._persist_clusters(response.clusters, batch_slice=batch_slice)

        self._log_prompt(
            batch_slice=batch_slice,
            prompt=prompt,
            response=json_data,
            latency_ms=latency_ms,
            token_estimate=token_estimate,
        )

        return ProposerResult(
            batch_id=batch_slice.batch_id,
            slice_id=batch_slice.slice_id,
            clusters=clusters,
            skipped_request_ids=response.skipped_request_ids,
            raw_response=json_data,
            latency_ms=round(latency_ms, 2),
            token_estimate=token_estimate,
        )

    def _build_prompt(
        self,
        batch_slice: BatchSlice,
        known_clusters_limit: int,
    ) -> RenderedPrompt:
        requests_payload = self._extract_requests(batch_slice.dataframe)
        known_clusters = self._known_clusters(limit=known_clusters_limit)
        max_clusters = self.settings.max_clusters_per_batch
        return render_cluster_proposer_prompt(
            batch_id=batch_slice.batch_id,
            requests=requests_payload,
            known_clusters=known_clusters,
            max_clusters=max_clusters,
            business_context=self.business_context,
        )

    def _persist_clusters(
        self,
        clusters: Sequence[ClusterProposal],
        batch_slice: BatchSlice,
    ) -> list[ClusterRecord]:
        """Persist validated cluster proposals to registry."""
        records: list[ClusterRecord] = []
        seen_ids: set[str] = set()

        for proposal in clusters:
            record = self._build_record_from_proposal(proposal, batch_slice)
            if record.cluster_id in seen_ids:
                record.cluster_id = self._deduplicate_cluster_id(record.cluster_id, seen_ids)
            seen_ids.add(record.cluster_id)
            records.append(record)

        if not records:
            return []

        return self.registry.bulk_upsert(records)

    def _build_record_from_proposal(
        self,
        proposal: ClusterProposal,
        batch_slice: BatchSlice,
    ) -> ClusterRecord:
        """Build ClusterRecord from validated ClusterProposal."""
        cluster_id = proposal.cluster_id
        if not cluster_id:
            cluster_id = self._generate_cluster_id(batch_slice.batch_id, proposal.name)
        
        # criteria может быть строкой или списком
        criteria = proposal.criteria
        if isinstance(criteria, list):
            criteria = "\n".join(f"- {item}" for item in criteria if item)
        else:
            criteria = str(criteria or "").strip()

        return ClusterRecord(
            cluster_id=cluster_id,
            name=proposal.name or cluster_id,
            summary=proposal.summary,
            criteria=criteria,
            sample_requests=proposal.sample_request_ids[:5],
            batch_id=batch_slice.batch_id,
            llm_reasoning=proposal.llm_reasoning or None,
        )

    @staticmethod
    def _generate_cluster_id(batch_id: str, name: str) -> str:
        base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "cluster"
        return f"{base}_{batch_id.split('-')[-1]}"

    @staticmethod
    def _deduplicate_cluster_id(cluster_id: str, existing: set[str]) -> str:
        idx = 2
        candidate = f"{cluster_id}_{idx}"
        while candidate in existing:
            idx += 1
            candidate = f"{cluster_id}_{idx}"
        return candidate

    @staticmethod
    def _extract_requests(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        records = dataframe.to_dict(orient="records")
        normalized: list[dict[str, Any]] = []
        for record in records:
            normalized.append(
                {
                    "request_id": record.get("request_id"),
                    "text": record.get("text_clean") or record.get("text"),
                    "text_raw": record.get("text_raw"),
                    "channel": record.get("channel"),
                    "priority": record.get("priority"),
                    "metadata": record.get("metadata"),
                }
            )
        return normalized

    def _known_clusters(self, limit: int) -> list[dict[str, Any]]:
        clusters = self.registry.list_clusters(limit=limit)
        return [
            {
                "cluster_id": record.cluster_id,
                "name": record.name,
                "summary": record.summary,
                "criteria": record.criteria,
                "sample_requests": record.sample_requests,
            }
            for record in clusters
        ]

    def _log_prompt(
        self,
        batch_slice: BatchSlice,
        prompt: RenderedPrompt,
        response: dict[str, Any],
        latency_ms: float,
        token_estimate: int,
    ) -> None:
        entry = self._create_log_entry(
            prompt_name="cluster_proposer_v0",
            batch_id=batch_slice.batch_id,
            slice_id=batch_slice.slice_id,
            prompt=prompt,
            response=response,
            latency_ms=latency_ms,
            metadata={
                "num_requests": len(batch_slice.dataframe.index),
                "max_clusters": self.settings.max_clusters_per_batch,
                "token_estimate": token_estimate,
            },
        )
        self.prompt_logger.log(entry)

