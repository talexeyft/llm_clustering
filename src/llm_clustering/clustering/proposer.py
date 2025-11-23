"""LLM Cluster Proposer implementation."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd
from loguru import logger

from llm_clustering.clustering.registry import ClusterRecord, ClusterRegistry
from llm_clustering.config import Settings, get_settings
from llm_clustering.llm import get_llm_provider
from llm_clustering.llm.base import BaseLLMProvider
from llm_clustering.llm.prompts import (
    PromptLogEntry,
    PromptLogger,
    RenderedPrompt,
    render_cluster_proposer_prompt,
)
from llm_clustering.pipeline import BatchSlice


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


class ClusterProposer:
    """Packages slice data, sends prompt to the LLM, and persists clusters."""

    def __init__(
        self,
        registry: ClusterRegistry | None = None,
        llm: BaseLLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or ClusterRegistry(settings=self.settings)
        self.llm = llm or get_llm_provider()
        self.prompt_logger = PromptLogger(self.settings)
        self.batch_config = self.settings.batch_config

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
        response_payload = self._parse_response(response_text, batch_slice.batch_id)

        clusters = self._persist_clusters(
            response_payload.get("clusters", []),
            batch_slice=batch_slice,
        )

        skipped_ids = response_payload.get("skipped_request_ids") or []
        if not isinstance(skipped_ids, list):
            skipped_ids = []

        self._log_prompt(
            batch_slice=batch_slice,
            prompt=prompt,
            response=response_payload,
            latency_ms=latency_ms,
            token_estimate=token_estimate,
        )

        return ProposerResult(
            batch_id=batch_slice.batch_id,
            slice_id=batch_slice.slice_id,
            clusters=clusters,
            skipped_request_ids=[str(item) for item in skipped_ids],
            raw_response=response_payload,
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
        max_clusters = self.batch_config.max_clusters_per_batch
        return render_cluster_proposer_prompt(
            batch_id=batch_slice.batch_id,
            requests=requests_payload,
            known_clusters=known_clusters,
            max_clusters=max_clusters,
        )

    def _execute_prompt(self, prompt: RenderedPrompt) -> tuple[str, float]:
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]
        start = time.perf_counter()
        response_text = self.llm.chat_completion(
            messages,
            temperature=self.settings.llm_config.temperature,
            max_tokens=self.settings.llm_config.max_tokens,
        )
        latency = time.perf_counter() - start
        return response_text, latency

    @staticmethod
    def _parse_response(response_text: str, batch_id: str) -> dict[str, Any]:
        from llm_clustering.clustering.utils import extract_json_from_response
        
        try:
            payload = extract_json_from_response(response_text)
        except json.JSONDecodeError as err:
            logger.error("Failed to parse proposer response for %s: %s", batch_id, err)
            logger.error("Original response: %s", response_text[:500])
            raise ValueError("LLM proposer returned invalid JSON.") from err

        if not isinstance(payload, dict):
            raise ValueError("LLM proposer returned non-object payload.")
        return payload

    def _persist_clusters(
        self,
        clusters_payload: Sequence[dict[str, Any]],
        batch_slice: BatchSlice,
    ) -> list[ClusterRecord]:
        records: list[ClusterRecord] = []
        seen_ids: set[str] = set()

        for payload in clusters_payload:
            record = self._build_record(payload, batch_slice)
            if record.cluster_id in seen_ids:
                record.cluster_id = self._deduplicate_cluster_id(record.cluster_id, seen_ids)
            seen_ids.add(record.cluster_id)
            records.append(record)

        if not records:
            return []

        return self.registry.bulk_upsert(records)

    def _build_record(
        self,
        payload: dict[str, Any],
        batch_slice: BatchSlice,
    ) -> ClusterRecord:
        cluster_id = self._sanitize_cluster_id(payload.get("cluster_id") or "")
        if not cluster_id:
            cluster_id = self._generate_cluster_id(batch_slice.batch_id, payload.get("name", "cluster"))

        sample_request_ids = payload.get("sample_request_ids") or []
        if not isinstance(sample_request_ids, list):
            sample_request_ids = []

        summary = str(payload.get("summary") or "").strip()
        
        # criteria может быть строкой или списком
        criteria_raw = payload.get("criteria")
        if isinstance(criteria_raw, list):
            criteria = "\n".join(f"- {item}" for item in criteria_raw if item)
        else:
            criteria = str(criteria_raw or "").strip()
        
        llm_reasoning = str(payload.get("llm_reasoning") or "").strip()
        name = str(payload.get("name") or cluster_id).strip()

        return ClusterRecord(
            cluster_id=cluster_id,
            name=name or cluster_id,
            summary=summary,
            criteria=criteria,
            sample_requests=[str(req_id) for req_id in sample_request_ids][:5],
            batch_id=batch_slice.batch_id,
            llm_reasoning=llm_reasoning or None,
        )

    @staticmethod
    def _sanitize_cluster_id(cluster_id: str) -> str:
        slug = re.sub(r"[^a-z0-9_]+", "_", cluster_id.strip().lower())
        slug = re.sub(r"_+", "_", slug).strip("_")
        return slug

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
        entry = PromptLogEntry(
            prompt_name="cluster_proposer_v0",
            batch_id=batch_slice.batch_id,
            slice_id=batch_slice.slice_id,
            prompt={"system": prompt.system, "user": prompt.user},
            response=response,
            latency_ms=round(latency_ms, 2),
            cost_estimate=None,
            metadata={
                "num_requests": len(batch_slice.dataframe.index),
                "max_clusters": self.batch_config.max_clusters_per_batch,
                "token_estimate": token_estimate,
            },
        )
        self.prompt_logger.log(entry)

    @staticmethod
    def _estimate_tokens(prompt: RenderedPrompt, response_text: str) -> int:
        total_chars = len(prompt.system) + len(prompt.user) + len(response_text)
        return max(1, total_chars // 4)

