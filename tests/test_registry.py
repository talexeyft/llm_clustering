"""Tests for ClusterRegistry persistence."""

from __future__ import annotations

from pathlib import Path

from llm_clustering.clustering.registry import ClusterRecord, ClusterRegistry
from llm_clustering.config.settings import Settings


def test_registry_persists_clusters(tmp_path: Path) -> None:
    storage_path = tmp_path / "registry.json"
    settings = Settings(
        batches_dir=tmp_path / "batches",
        results_dir=tmp_path / "results",
        reports_dir=tmp_path / "reports",
        log_file=str(tmp_path / "clustering.log"),
        log_prompt_dir=str(tmp_path / "prompts"),
        metrics_file=str(tmp_path / "metrics.csv"),
    )
    registry = ClusterRegistry(storage_path=storage_path, settings=settings)
    record = ClusterRecord(
        cluster_id="billing_issue",
        name="Оплата",
        summary="Проблемы с оплатой заказов",
        criteria="Любые вопросы по оплате",
    )

    registry.upsert(record)
    registry.record_assignment("billing_issue", "req-1")

    reloaded = ClusterRegistry(storage_path=storage_path, settings=settings)
    loaded_record = reloaded.get_cluster("billing_issue")

    assert loaded_record is not None
    assert loaded_record.count == 1
    assert "req-1" in loaded_record.sample_requests

