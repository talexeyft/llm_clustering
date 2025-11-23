"""LLM Assignment Judge implementation."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence, cast

from loguru import logger

from llm_clustering.clustering.base_llm_component import BaseLLMComponent
from llm_clustering.clustering.registry import ClusterRecord, ClusterRegistry
from llm_clustering.clustering.schemas import JudgeResponse, SuggestedCluster
from llm_clustering.config import Settings
from llm_clustering.llm.base import BaseLLMProvider
from llm_clustering.llm.prompts import (
    RenderedPrompt,
    render_assignment_judge_prompt,
)
from llm_clustering.pipeline.batch_builder import BatchSlice

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


class AssignmentJudge(BaseLLMComponent):
    """Evaluates each request against the registry and batch clusters."""

    def __init__(
        self,
        registry: ClusterRegistry | None = None,
        llm: BaseLLMProvider | None = None,
        settings: Settings | None = None,
        business_context: str | None = None,
    ) -> None:
        super().__init__(registry, llm, settings, business_context)

    def judge_slice(
        self,
        batch_slice: BatchSlice,
        candidate_clusters: Sequence[ClusterRecord] | None = None,
        parallel_batch_size: int | None = None,
    ) -> list[AssignmentResult]:
        """Evaluate every row inside the slice with parallel processing.
        
        Args:
            batch_slice: Batch slice to process
            candidate_clusters: Candidate clusters for assignment
            parallel_batch_size: Number of parallel requests to process 
                                (defaults to value from config)
        """
        dataframe = batch_slice.dataframe
        if candidate_clusters is None:
            candidate_clusters = self.registry.list_clusters(limit=20)

        # Use config value if not explicitly provided
        if parallel_batch_size is None:
            parallel_batch_size = self.settings.parallel_batch_size

        cluster_payload = self._format_clusters(candidate_clusters)
        records = dataframe.to_dict(orient="records")
        results: list[AssignmentResult] = []

        # Process records in parallel batches
        logger.info(f"Processing {len(records)} records with parallel batch size {parallel_batch_size}")
        
        with ThreadPoolExecutor(max_workers=parallel_batch_size) as executor:
            # Submit all tasks
            future_to_record = {
                executor.submit(
                    self._process_single_record,
                    batch_slice,
                    cluster_payload,
                    record,
                    candidate_clusters,
                ): record
                for record in records
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_record):
                record = future_to_record[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing record {record.get('request_id')}: {e}")
                    # Create error result
                    results.append(
                        AssignmentResult(
                            batch_id=batch_slice.batch_id,
                            slice_id=batch_slice.slice_id,
                            request_id=str(record.get("request_id")),
                            decision="skip",
                            cluster_id=None,
                            confidence_text="low (error)",
                            llm_rationale=f"Ошибка обработки: {str(e)}",
                            raw_response={},
                            latency_ms=0.0,
                            token_estimate=0,
                        )
                    )

        logger.info(f"Completed processing {len(results)} records")
        return results
    
    def _process_single_record(
        self,
        batch_slice: BatchSlice,
        cluster_payload: list[dict[str, Any]],
        record: dict[str, Any],
        candidate_clusters: Sequence[ClusterRecord],
    ) -> AssignmentResult:
        """Process a single record (used for parallel execution)."""
        prompt = self._build_prompt(batch_slice.batch_id, cluster_payload, record)
        response_text, latency = self._execute_prompt(prompt)
        latency_ms = latency * 1000

        try:
            json_data = self._parse_json_response(response_text, "judge")
            judge_response = JudgeResponse.model_validate(json_data)
        except (json.JSONDecodeError, ValueError) as err:
            logger.error("Judge returned invalid response for %s: %s", record.get("request_id"), err)
            return AssignmentResult(
                batch_id=batch_slice.batch_id,
                slice_id=batch_slice.slice_id,
                request_id=str(record.get("request_id")),
                decision="skip",
                cluster_id=None,
                confidence_text="low (invalid response)",
                llm_rationale="Ответ модели нельзя разобрать.",
                raw_response={},
                latency_ms=round(latency_ms, 2),
                token_estimate=0,
            )
        except Exception as err:
            logger.error("Failed to validate judge response for %s: %s", record.get("request_id"), err)
            return AssignmentResult(
                batch_id=batch_slice.batch_id,
                slice_id=batch_slice.slice_id,
                request_id=str(record.get("request_id")),
                decision="skip",
                cluster_id=None,
                confidence_text="low (validation error)",
                llm_rationale="Ошибка валидации ответа.",
                raw_response={},
                latency_ms=round(latency_ms, 2),
                token_estimate=0,
            )

        result = self._build_result_from_response(
            batch_slice=batch_slice,
            judge_response=judge_response,
            cluster_context=candidate_clusters,
            latency_ms=latency_ms,
            token_estimate=self._estimate_tokens(prompt, response_text),
        )

        self._log_prompt(
            batch_slice=batch_slice,
            request_id=result.request_id,
            prompt=prompt,
            response=json_data,
            latency_ms=latency_ms,
        )

        return result

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
            business_context=self.business_context,
        )

    def _build_result_from_response(
        self,
        batch_slice: BatchSlice,
        judge_response: JudgeResponse,
        cluster_context: Sequence[ClusterRecord],
        latency_ms: float,
        token_estimate: int,
    ) -> AssignmentResult:
        """Build AssignmentResult from validated JudgeResponse."""
        decision = judge_response.decision
        decision_literal = cast(DecisionLiteral, decision)
        request_id = judge_response.request_id
        cluster_id = judge_response.cluster_id or None
        
        suggested_cluster = None
        if decision == "assign" and cluster_id:
            try:
                self.registry.record_assignment(cluster_id, request_id)
            except KeyError:
                logger.warning("Cluster %s missing; downgrading decision to skip.", cluster_id)
                decision = "skip"
                decision_literal = "skip"
                cluster_id = None
        elif decision == "new_cluster" and judge_response.suggested_cluster:
            suggested_cluster = self._convert_suggested_to_cluster(
                judge_response.suggested_cluster,
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
            confidence_text=judge_response.confidence_text,
            llm_rationale=judge_response.llm_rationale,
            raw_response=judge_response.model_dump(),
            latency_ms=round(latency_ms, 2),
            token_estimate=token_estimate,
            suggested_cluster=suggested_cluster,
        )

    def _convert_suggested_to_cluster(
        self,
        suggested: SuggestedCluster,
        batch_slice: BatchSlice,
        request_id: str,
    ) -> ClusterRecord | None:
        """Convert validated SuggestedCluster to ClusterRecord."""
        if not suggested.name or not suggested.summary:
            return None

        cluster_id = suggested.cluster_id or suggested.name
        slug = self._sanitize_cluster_id(cluster_id)
        
        sample_requests = list(suggested.sample_request_ids)
        if request_id not in sample_requests:
            sample_requests.insert(0, request_id)

        return ClusterRecord(
            cluster_id=slug or self._fallback_cluster_id(suggested.name, batch_slice.slice_id),
            name=suggested.name,
            summary=suggested.summary,
            criteria=suggested.criteria,
            sample_requests=sample_requests[:5],
            batch_id=batch_slice.batch_id,
            status="tentative",
            llm_reasoning=suggested.llm_rationale or "",
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
        response_text = json.dumps(response, ensure_ascii=False)
        entry = self._create_log_entry(
            prompt_name="assignment_judge_v0",
            batch_id=batch_slice.batch_id,
            slice_id=batch_slice.slice_id,
            prompt=prompt,
            response=response,
            latency_ms=latency_ms,
            metadata={
                "request_id": request_id,
                "token_estimate": self._estimate_tokens(prompt, response_text),
            },
        )
        self.prompt_logger.log(entry)

