"""Utilities for preparing deterministic LLM batches."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pandas as pd
from loguru import logger

from llm_clustering.config import Settings, get_settings

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class SnapshotPaths:
    """Locations of the persisted batch artifacts."""

    csv: Path
    parquet: Path | None


@dataclass(slots=True)
class BatchSlice:
    """Single deterministic slice within a batch."""

    batch_id: str
    slice_id: str
    dataframe: pd.DataFrame
    snapshot: SnapshotPaths

    @property
    def size(self) -> int:
        """Number of rows inside the slice."""
        return len(self.dataframe.index)


@dataclass(slots=True)
class BatchBuildResult:
    """Full result of the batch preparation phase."""

    batch_id: str
    prepared: pd.DataFrame
    prepared_snapshot: SnapshotPaths
    slices: list[BatchSlice]

    @property
    def total_rows(self) -> int:
        """Total number of rows in the prepared DataFrame."""
        return len(self.prepared.index)


class BatchBuilder:
    """Normalize raw requests and split them into LLM-friendly slices."""

    def __init__(
        self,
        settings: Settings | None = None,
        text_column: str = "text",
    ) -> None:
        self.settings = settings or get_settings()
        self.text_column = text_column
        self.batches_dir = Path(self.settings.batches_dir)
        self.batches_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        dataframe: pd.DataFrame,
        batch_id: str | None = None,
    ) -> BatchBuildResult:
        """Prepare dataframe, persist snapshot, and emit deterministic slices."""
        prepared_df, batch_id, prepared_snapshot = self._prepare_dataframe(
            dataframe, batch_id=batch_id
        )
        slices = list(self._iter_slices(prepared_df, batch_id=batch_id))
        return BatchBuildResult(
            batch_id=batch_id,
            prepared=prepared_df,
            prepared_snapshot=prepared_snapshot,
            slices=slices,
        )

    def _prepare_dataframe(
        self,
        dataframe: pd.DataFrame,
        batch_id: str | None = None,
    ) -> tuple[pd.DataFrame, str, SnapshotPaths]:
        if not isinstance(dataframe, pd.DataFrame):
            msg = "BatchBuilder expects a pandas.DataFrame."
            raise TypeError(msg)

        # Check for text column
        if self.text_column not in dataframe.columns:
            msg = f"DataFrame is missing required text column: {self.text_column}"
            raise ValueError(msg)

        working_df = dataframe.copy()
        resolved_batch_id = batch_id or self._generate_batch_id()

        # Auto-generate request_id if not present
        if "request_id" not in working_df.columns:
            working_df["request_id"] = [f"req-{resolved_batch_id}-{i}" for i in range(len(working_df))]
        
        working_df["batch_id"] = resolved_batch_id
        working_df["request_id"] = working_df["request_id"].astype(str)
        working_df["text_raw"] = working_df[self.text_column].fillna("").astype(str)
        working_df["text_clean"] = working_df["text_raw"].map(self._normalize_text)
        working_df["batch_row_index"] = range(len(working_df.index))

        snapshot = self._persist_snapshot(
            working_df,
            file_stem=f"{resolved_batch_id}_prepared",
        )

        snapshot_info = snapshot.csv.name if snapshot.csv else "not saved"
        logger.info(
            "Prepared batch %s with %d rows (snapshot: %s)",
            resolved_batch_id,
            len(working_df.index),
            snapshot_info,
        )

        return working_df, resolved_batch_id, snapshot

    def _iter_slices(
        self,
        dataframe: pd.DataFrame,
        batch_id: str,
    ) -> Iterator[BatchSlice]:
        batch_size = self.settings.batch_size
        total_rows = len(dataframe.index)

        if total_rows == 0:
            return

        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            slice_idx = start // batch_size
            slice_id = f"{batch_id}-{slice_idx:04d}"
            slice_df = dataframe.iloc[start:end].copy()
            slice_df["batch_slice_id"] = slice_id

            snapshot = self._persist_snapshot(
                slice_df,
                file_stem=f"{batch_id}_slice_{slice_idx:04d}",
            )

            snapshot_info = snapshot.csv.name if snapshot.csv else "not saved"
            logger.debug(
                "Created slice %s with %d rows (snapshot: %s)",
                slice_id,
                len(slice_df.index),
                snapshot_info,
            )

            yield BatchSlice(
                batch_id=batch_id,
                slice_id=slice_id,
                dataframe=slice_df,
                snapshot=snapshot,
            )

    def _persist_snapshot(
        self,
        dataframe: pd.DataFrame,
        file_stem: str,
    ) -> SnapshotPaths:
        # Check if we should save based on file_stem
        is_slice = "_slice_" in file_stem
        should_save = (
            (is_slice and self.settings.save_slices) or
            (not is_slice and self.settings.save_batches)
        )
        
        if not should_save:
            # Return empty paths without saving
            return SnapshotPaths(csv=None, parquet=None)
        
        csv_path = self.batches_dir / f"{file_stem}.csv"
        parquet_path = self.batches_dir / f"{file_stem}.parquet"

        dataframe.to_csv(csv_path, index=False)

        parquet_written = True
        try:
            dataframe.to_parquet(parquet_path, index=False)
        except (ImportError, ValueError) as exc:
            parquet_written = False
            logger.warning(
                "Parquet snapshot skipped for %s (%s). Install pyarrow/fastparquet to enable it.",
                parquet_path.name,
                exc,
            )

        return SnapshotPaths(
            csv=csv_path,
            parquet=parquet_path if parquet_written else None,
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        unescaped = html.unescape(text)
        stripped_tags = HTML_TAG_RE.sub(" ", unescaped)
        normalized = WHITESPACE_RE.sub(" ", stripped_tags)
        return normalized.strip()

    @staticmethod
    def _generate_batch_id() -> str:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
        random_suffix = uuid4().hex[:6]
        return f"batch-{timestamp}-{random_suffix}"

