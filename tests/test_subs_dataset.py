"""Tests for the subs parquet sample helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_clustering.data.subs_dataset import (
    load_subs_messages,
    materialize_subs_sample,
)


def _write_parquet(path: Path, rows: list[dict[str, str]]) -> None:
    dataframe = pd.DataFrame(rows)
    dataframe.to_parquet(path, index=False)


def test_load_subs_messages_reads_expected_rows(tmp_path: Path) -> None:
    subs_dir = tmp_path / "subs"
    subs_dir.mkdir()

    _write_parquet(
        subs_dir / "0000.parquet",
        [
            {
                "conversation_id": "conv-1",
                "speaker": "customer",
                "date_time": "2024-10-01T09:00:00",
                "text": "First message",
            },
            {
                "conversation_id": "conv-1",
                "speaker": "agent",
                "date_time": "2024-10-01T09:00:10",
                "text": "Second message",
            },
        ],
    )
    _write_parquet(
        subs_dir / "0001.parquet",
        [
            {
                "conversation_id": "conv-2",
                "speaker": "customer",
                "date_time": "2024-10-02T10:00:00",
                "text": "Third message",
            },
            {
                "conversation_id": "conv-3",
                "speaker": "agent",
                "date_time": "2024-10-03T11:00:00",
                "text": "Fourth message",
            },
        ],
    )

    dataframe = load_subs_messages(sample_size=3, subs_dir=subs_dir)

    assert len(dataframe.index) == 3
    assert dataframe["request_id"].is_unique
    assert dataframe.iloc[0]["text"] == "First message"
    assert dataframe.iloc[2]["conversation_id"] == "conv-2"


def test_materialize_subs_sample_creates_files(tmp_path: Path) -> None:
    subs_dir = tmp_path / "subs"
    subs_dir.mkdir()

    _write_parquet(
        subs_dir / "0000.parquet",
        [
            {
                "conversation_id": "conv-x",
                "speaker": "customer",
                "date_time": "2024-09-01T08:00:00",
                "text": "Hello there",
            },
            {
                "conversation_id": "conv-y",
                "speaker": "agent",
                "date_time": "2024-09-01T08:05:00",
                "text": "We can help",
            },
        ],
    )

    output_dir = tmp_path / "out"
    result = materialize_subs_sample(
        sample_size=2,
        subs_dir=subs_dir,
        output_dir=output_dir,
        file_stem="demo",
    )

    assert result.parquet_path.exists()
    assert result.csv_path.exists()

    saved = pd.read_parquet(result.parquet_path)
    assert "request_id" in saved.columns
    assert saved.iloc[0]["text"] == "Hello there"


