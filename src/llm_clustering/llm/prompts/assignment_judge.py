"""Prompt template for Assignment Judge stage."""

from __future__ import annotations

from textwrap import dedent
from typing import Iterable, Sequence

from llm_clustering.llm.prompts.template import PromptTemplate, RenderedPrompt

ASSIGNMENT_JUDGE_TEMPLATE = PromptTemplate(
    name="assignment_judge_v0",
    system=dedent(
        """
        /nothink
        
        You decide whether a single contact-center request belongs to one of the
        provided clusters. Answer conservatively, use Russian, and never assign
        when confidence is low.
        
        CRITICAL: Do NOT use <think> tags or any reasoning markup. 
        Reply ONLY with valid JSON, no text before or after.
        """
    ),
    user=dedent(
        """
        ## Batch meta
        batch_id: $batch_id

        ## Request
        id: $request_id
        text: $request_text

        ## Candidate clusters
        $candidate_clusters

        Return JSON with schema:
        {
          "request_id": "$request_id",
          "decision": "assign|new_cluster|skip",
          "cluster_id": "cluster_id or empty string",
          "confidence_text": "low|medium|high + justification",
          "llm_rationale": "2 sentences in Russian",
          "suggested_cluster": {
            "name": "",
            "summary": "",
            "criteria": "",
            "sample_request_ids": []
          }
        }

        Rules:
        - decision=assign only when the request clearly matches one cluster
        - decision=new_cluster when you see a strong new theme (fill suggested_cluster)
        - decision=skip for unclear or insufficient data
        """
    ),
)


def render_assignment_judge_prompt(
    batch_id: str,
    request: dict[str, str],
    candidate_clusters: Sequence[dict[str, str]],
) -> RenderedPrompt:
    """Render judge prompt for a single request."""
    context = {
        "batch_id": batch_id,
        "request_id": request.get("request_id", "unknown"),
        "request_text": request.get("text_clean") or request.get("text") or "",
        "candidate_clusters": _format_candidate_clusters(candidate_clusters),
    }
    return ASSIGNMENT_JUDGE_TEMPLATE.render(context)


def _format_candidate_clusters(clusters: Iterable[dict[str, str]]) -> str:
    formatted: list[str] = []
    for cluster in clusters:
        cluster_id = cluster.get("cluster_id", "unknown_cluster")
        name = cluster.get("name", "")
        summary = cluster.get("summary", "")
        criteria = cluster.get("criteria", "")
        formatted.append(
            dedent(
                f"""
                - {cluster_id} :: {name}
                  summary: {summary}
                  criteria: {criteria}
                """
            ).strip()
        )
    return "\n".join(formatted) if formatted else "Нет доступных кластеров."

