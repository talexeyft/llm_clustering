"""LLM-driven sanity checks for cluster cohesion."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from llm_clustering.config import Settings, get_settings
from llm_clustering.llm import get_llm_provider
from llm_clustering.llm.base import BaseLLMProvider
from llm_clustering.llm.prompts import PromptLogEntry, PromptLogger


class CohesionChecker:
    """Asks the LLM to rate cohesion for top clusters in a batch."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: BaseLLMProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or get_llm_provider()
        self.reports_dir = Path(self.settings.reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_logger = PromptLogger(self.settings)

    def run(
        self,
        batch_id: str,
        assignments: pd.DataFrame,
        prepared_df: pd.DataFrame,
        top_n: int = 5,
    ) -> Path | None:
        if assignments.empty:
            return None

        summary = self._build_summary(assignments, prepared_df, top_n=top_n)
        if not summary:
            return None

        prompt = self._render_prompt(batch_id, summary)
        messages = [
            {"role": "system", "content": "Ты аналитик качества. Ответь на русском."},
            {"role": "user", "content": prompt},
        ]

        start = time.perf_counter()
        response = self.llm.chat_completion(messages, temperature=0.1, max_tokens=512)
        latency_ms = (time.perf_counter() - start) * 1000

        report_path = self.reports_dir / f"{batch_id}_cohesion.txt"
        report_path.write_text(response.strip(), encoding="utf-8")

        self.prompt_logger.log(
            PromptLogEntry(
                prompt_name="cohesion_check_v0",
                batch_id=batch_id,
                slice_id=None,
                prompt={"system": messages[0]["content"], "user": prompt},
                response={"text": response},
                latency_ms=round(latency_ms, 2),
                cost_estimate=None,
                metadata={"clusters_evaluated": len(summary)},
            )
        )

        return report_path

    @staticmethod
    def _build_summary(
        assignments: pd.DataFrame,
        prepared_df: pd.DataFrame,
        top_n: int,
    ) -> list[dict[str, Any]]:
        merged = assignments.merge(
            prepared_df[["request_id", "text_clean", "text_raw"]],
            on="request_id",
            how="left",
        )
        filtered = merged[merged["cluster_id"].notna()]
        if filtered.empty:
            return []

        grouped = filtered.groupby("cluster_id")
        summary: list[dict[str, Any]] = []
        for cluster_id, frame in grouped:
            texts = (
                frame["text_clean"]
                .fillna(frame["text_raw"])
                .dropna()
                .astype(str)
                .tolist()
            )
            summary.append(
                {
                    "cluster_id": cluster_id,
                    "count": len(frame.index),
                    "samples": texts[:3],
                }
            )

        summary.sort(key=lambda item: item["count"], reverse=True)
        return summary[:top_n]

    @staticmethod
    def _render_prompt(batch_id: str, summary: list[dict[str, Any]]) -> str:
        lines = [
            f"Batch: {batch_id}",
            "Для каждого кластера оцени однородность обращений как low/medium/high и объясни почему.",
            "Формат ответа: JSON список объектов {cluster_id, cohesion, reason}.",
            "",
        ]
        for item in summary:
            lines.append(f"Кластер {item['cluster_id']} (n={item['count']}):")
            for sample in item["samples"]:
                lines.append(f"- {sample}")
            lines.append("")
        return "\n".join(lines).strip()

