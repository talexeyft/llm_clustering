"""Prompt template for the LLM Cluster Proposer stage."""

from __future__ import annotations

from textwrap import dedent
from typing import Iterable, Sequence

from llm_clustering.llm.prompts.template import PromptTemplate, RenderedPrompt

CLUSTER_PROPOSER_TEMPLATE = PromptTemplate(
    name="cluster_proposer_v0",
    system=dedent(
        """
        You are an expert support analyst who clusters contact-center requests using
        only the given data. You think in Russian, produce concise reasoning,
        and NEVER invent additional facts. Respect the batch boundaries.
        """
    ),
    user=dedent(
        """
        ## Batch meta
        batch_id: $batch_id
        max_clusters: $max_clusters

        ## Historical clusters
        $known_clusters

        ## Incoming requests
        $requests_block

        Return ONLY valid JSON with the schema:
        {
          "batch_id": "$batch_id",
          "clusters": [
            {
              "cluster_id": "string (snake_case, unique inside registry)",
              "name": "short title (<=8 words)",
              "summary": "what is the issue and why it matters",
              "criteria": "inclusion/exclusion criteria as bullet text",
              "sample_request_ids": ["req-1","req-2"],
              "llm_reasoning": "2-3 sentences explaining why the cluster exists"
            }
          ],
          "skipped_request_ids": ["req-x", "..."]
        }

        Rules:
        - propose up to $max_clusters clusters
        - keep sample_request_ids to 3-5 per cluster
        - if requests are too diverse put their ids into skipped_request_ids
        - always write output in Russian
        """
    ),
)


def render_cluster_proposer_prompt(
    batch_id: str,
    requests: Sequence[dict[str, str]],
    known_clusters: Sequence[dict[str, str]] | None,
    max_clusters: int,
) -> RenderedPrompt:
    """Render the proposer prompt with contextual data."""
    context = {
        "batch_id": batch_id,
        "max_clusters": max_clusters,
        "known_clusters": _format_known_clusters(known_clusters or []),
        "requests_block": _format_requests(requests),
    }
    return CLUSTER_PROPOSER_TEMPLATE.render(context)


def _format_requests(requests: Sequence[dict[str, str]]) -> str:
    if not requests:
        return "Нет новых обращений в этом батче."

    lines: list[str] = []
    for idx, req in enumerate(requests, start=1):
        request_id = req.get("request_id", f"req-{idx}")
        text = req.get("text") or req.get("text_clean") or req.get("text_raw") or ""
        channel = req.get("channel", "")
        priority = req.get("priority", "")
        meta_suffix = " ".join(
            filter(
                None,
                [
                    f"channel={channel}" if channel else "",
                    f"priority={priority}" if priority else "",
                ],
            )
        )
        lines.append(f"- [{request_id}] {text.strip()} {meta_suffix}".strip())
    return "\n".join(lines)


def _format_known_clusters(clusters: Iterable[dict[str, str]]) -> str:
    formatted: list[str] = []
    for cluster in clusters:
        cluster_id = cluster.get("cluster_id", "unknown_cluster")
        name = cluster.get("name", "Без названия")
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

    return "\n".join(formatted) if formatted else "Нет подтвержденных кластеров. Создай новые, если есть устойчивые темы."

