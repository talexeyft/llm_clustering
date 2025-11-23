"""LLM Assignment Judge implementation."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence, cast

from loguru import logger

from llm_clustering.clustering.registry import ClusterRecord, ClusterRegistry
from llm_clustering.config import Settings, get_settings
from llm_clustering.llm import get_llm_provider
from llm_clustering.llm.base import BaseLLMProvider
from llm_clustering.llm.prompts import (
    PromptLogEntry,
    PromptLogger,
    RenderedPrompt,
    render_assignment_judge_prompt,
)
from llm_clustering.pipeline import BatchSlice

DecisionLiteral = Literal["assign", "new_cluster", "skip"]


@dataclass(slots=True)
class AssignmentResult:
    """Decision emitted by Assignment Judge."""

    batch_id: str
    slice_id: str
    request_id: str
    decision: DecisionLiteral
    cluster_id: str | None
    confidence_text: str
    llm_rationale: str
    raw_response: dict[str, Any]
    latency_ms: float
    token_estimate: int
    suggested_cluster: ClusterRecord | None = None


class AssignmentJudge:
    """Evaluates each request against the registry and batch clusters."""

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

    def judge_slice(
        self,
        batch_slice: BatchSlice,
        candidate_clusters: Sequence[ClusterRecord] | None = None,
    ) -> list[AssignmentResult]:
        """Evaluate every row inside the slice."""
        dataframe = batch_slice.dataframe
        if candidate_clusters is None:
            candidate_clusters = self.registry.list_clusters(limit=20)

        cluster_payload = self._format_clusters(candidate_clusters)
        results: list[AssignmentResult] = []

        for record in dataframe.to_dict(orient="records"):
            prompt = self._build_prompt(batch_slice.batch_id, cluster_payload, record)
            response_text, latency = self._execute_prompt(prompt)
            latency_ms = latency * 1000

            try:
                from llm_clustering.clustering.utils import extract_json_from_response
                response_payload = extract_json_from_response(response_text)
                if not isinstance(response_payload, dict):
                    raise ValueError("Judge response must be a JSON object.")
            except json.JSONDecodeError as err:
                logger.error("Judge returned invalid JSON for %s: %s", record.get("request_id"), err)
                logger.error("Original response: %s", response_text[:500])
                results.append(
                    AssignmentResult(
                        batch_id=batch_slice.batch_id,
                        slice_id=batch_slice.slice_id,
                        request_id=str(record.get("request_id")),
                        decision="skip",
                        cluster_id=None,
                        confidence_text="low (invalid JSON)",
                        llm_rationale="Ответ модели нельзя разобрать.",
                        raw_response={},
                        latency_ms=round(latency_ms, 2),
                        token_estimate=0,
                    )
                )
                continue

            result = self._build_result(
                batch_slice=batch_slice,
                request_record=record,
                response=response_payload,
                cluster_context=candidate_clusters,
                latency_ms=latency_ms,
                token_estimate=self._estimate_tokens(prompt, response_text),
            )
            results.append(result)

            self._log_prompt(
                batch_slice=batch_slice,
                request_id=result.request_id,
                prompt=prompt,
                response=response_payload,
                latency_ms=latency_ms,
            )

        return results

    def _build_prompt(
        self,
        batch_id: str,
        candidate_clusters: list[dict[str, Any]],
        request_record: dict[str, Any],
    ) -> RenderedPrompt:
        return render_assignment_judge_prompt(
            batch_id=batch_id,
            request=request_record,
            candidate_clusters=candidate_clusters,
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

    def _build_result(
        self,
        batch_slice: BatchSlice,
        request_record: dict[str, Any],
        response: dict[str, Any],
        cluster_context: Sequence[ClusterRecord],
        latency_ms: float,
        token_estimate: int,
    ) -> AssignmentResult:
        decision = str(response.get("decision") or "skip").strip().lower()
        if decision not in {"assign", "new_cluster", "skip"}:
            decision = "skip"
        decision_literal = cast(DecisionLiteral, decision)

        request_id = str(response.get("request_id") or request_record.get("request_id"))
        cluster_id = response.get("cluster_id")
        cluster_id = str(cluster_id).strip() if cluster_id else None
        confidence_text = str(response.get("confidence_text") or "low").strip()
        rationale = str(response.get("llm_rationale") or "").strip()
        suggested_payload = response.get("suggested_cluster") or {}

        suggested_cluster = None
        if decision == "assign" and cluster_id:
            try:
                self.registry.record_assignment(cluster_id, request_id)
            except KeyError:
                logger.warning("Cluster %s missing; downgrading decision to skip.", cluster_id)
                decision = "skip"
                cluster_id = None
        elif decision == "new_cluster" and suggested_payload:
            suggested_cluster = self._convert_to_cluster(
                suggested_payload,
                batch_slice=batch_slice,
                request_id=request_id,
            )
            if suggested_cluster:
                self.registry.upsert(suggested_cluster)
                cluster_id = suggested_cluster.cluster_id
        else:
            cluster_id = None if decision != "assign" else cluster_id

        return AssignmentResult(
            batch_id=batch_slice.batch_id,
            slice_id=batch_slice.slice_id,
            request_id=request_id,
            decision=decision_literal,
            cluster_id=cluster_id,
            confidence_text=confidence_text,
            llm_rationale=rationale,
            raw_response=response,
             latency_ms=round(latency_ms, 2),
             token_estimate=token_estimate,
            suggested_cluster=suggested_cluster,
        )

    def _convert_to_cluster(
        self,
        payload: dict[str, Any],
        batch_slice: BatchSlice,
        request_id: str,
    ) -> ClusterRecord | None:
        name = str(payload.get("name") or "").strip()
        summary = str(payload.get("summary") or "").strip()
        if not name or not summary:
            return None

        cluster_id = payload.get("cluster_id") or name
        slug = self._sanitize_cluster_id(cluster_id)
        sample_requests = payload.get("sample_request_ids") or []
        if not isinstance(sample_requests, list):
            sample_requests = []
        if request_id not in sample_requests:
            sample_requests.insert(0, request_id)

        return ClusterRecord(
            cluster_id=slug or self._fallback_cluster_id(name, batch_slice.slice_id),
            name=name,
            summary=summary,
            criteria=str(payload.get("criteria") or ""),
            sample_requests=[str(item) for item in sample_requests][:5],
            batch_id=batch_slice.batch_id,
            status="tentative",
            llm_reasoning=str(payload.get("llm_rationale") or ""),
        )

    @staticmethod
    def _fallback_cluster_id(name: str, slice_id: str) -> str:
        base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "cluster"
        return f"{base}_{slice_id.split('-')[-1]}"

    @staticmethod
    def _sanitize_cluster_id(cluster_id: str | None) -> str:
        if not cluster_id:
            return ""
        slug = re.sub(r"[^a-z0-9_]+", "_", str(cluster_id).lower())
        slug = re.sub(r"_+", "_", slug).strip("_")
        return slug

    def _format_clusters(self, clusters: Iterable[ClusterRecord]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for record in clusters:
            formatted.append(
                {
                    "cluster_id": record.cluster_id,
                    "name": record.name,
                    "summary": record.summary,
                    "criteria": record.criteria,
                }
            )
        return formatted

    def _log_prompt(
        self,
        batch_slice: BatchSlice,
        request_id: str,
        prompt: RenderedPrompt,
        response: dict[str, Any],
        latency_ms: float,
    ) -> None:
        entry = PromptLogEntry(
            prompt_name="assignment_judge_v0",
            batch_id=batch_slice.batch_id,
            slice_id=batch_slice.slice_id,
            prompt={"system": prompt.system, "user": prompt.user},
            response=response,
            latency_ms=round(latency_ms, 2),
            cost_estimate=None,
            metadata={
                "request_id": request_id,
                "token_estimate": self._estimate_tokens(prompt, json.dumps(response, ensure_ascii=False)),
            },
        )
        self.prompt_logger.log(entry)

    @staticmethod
    def _estimate_tokens(prompt: RenderedPrompt, response_text: str) -> int:
        total_chars = len(prompt.system) + len(prompt.user) + len(response_text)
        return max(1, total_chars // 4)

