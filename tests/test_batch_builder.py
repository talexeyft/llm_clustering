"""Tests for BatchBuilder data preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_clustering.config.settings import Settings
from llm_clustering.pipeline import BatchBuilder


def test_batch_builder_creates_clean_columns(tmp_path: Path) -> None:
    settings = Settings(
        batches_dir=tmp_path / "batches",
        results_dir=tmp_path / "results",
        reports_dir=tmp_path / "reports",
        log_file=str(tmp_path / "clustering.log"),
        log_prompt_dir=str(tmp_path / "prompts"),
        metrics_file=str(tmp_path / "metrics.csv"),
    )
    builder = BatchBuilder(settings=settings, text_column="text")

    dataframe = pd.DataFrame(
        {
            "request_id": ["req-1", "req-2"],
            "text": ["<b>Hello</b>   world", "   Second\nline   "],
        }
    )

    build_result = builder.build(dataframe, batch_id="batch-test")

    assert "text_clean" in build_result.prepared.columns
    assert build_result.prepared_snapshot.csv.exists()
    assert len(build_result.slices) == 1
    first_slice = build_result.slices[0]
    assert first_slice.snapshot.csv.exists()
    assert first_slice.dataframe["text_clean"].tolist()[0] == "Hello world"

