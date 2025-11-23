"""Helpers for working with the ~/data/subs parquet dumps."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from loguru import logger

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - pyarrow is optional at runtime
    pq = None

DEFAULT_SUBS_DIR = Path.home() / "data" / "subs"
DEFAULT_COLUMNS = ["conversation_id", "speaker", "date_time", "text"]
DEFAULT_SAMPLE_SIZE = 1_000
DEFAULT_BATCH_SIZE = 2_000
DEFAULT_OUTPUT_DIR = Path("ai_data")
DEFAULT_FILE_STEM = "subs_sample"


@dataclass(slots=True)
class SubsSampleResult:
    """Result of materializing a parquet/csv sample."""

    dataframe: pd.DataFrame
    parquet_path: Path
    csv_path: Path


def load_subs_messages(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    subs_dir: Path | str = DEFAULT_SUBS_DIR,
    columns: Sequence[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Return the first `sample_size` messages as a pandas.DataFrame.

    The function streams from multiple parquet files if needed and normalizes
    the schema expected by the clustering pipeline (request_id + text column).
    """

    if sample_size < 0:
        msg = "sample_size must be >= 0"
        raise ValueError(msg)

    subs_path = Path(subs_dir).expanduser()
    if not subs_path.exists():
        msg = f"Subs directory does not exist: {subs_path}"
        raise FileNotFoundError(msg)

    parquet_files = sorted(subs_path.glob("*.parquet"))
    if not parquet_files:
        msg = f"No parquet files found in {subs_path}"
        raise FileNotFoundError(msg)

    cols = list(columns) if columns else DEFAULT_COLUMNS
    remaining = sample_size or float("inf")
    collected_frames: list[pd.DataFrame] = []

    for file_path in parquet_files:
        if remaining <= 0:
            break

        rows_to_take = int(min(remaining, 10_000)) if sample_size else 10_000
        chunk = _read_parquet_slice(
            file_path=file_path,
            rows_to_take=rows_to_take,
            columns=cols,
            batch_size=batch_size,
        )

        if chunk.empty:
            continue

        collected_frames.append(chunk)
        if sample_size:
            remaining -= len(chunk.index)

    if not collected_frames:
        msg = "Failed to read any rows from subs parquet files."
        raise RuntimeError(msg)

    dataframe = pd.concat(collected_frames, ignore_index=True)
    if sample_size:
        dataframe = dataframe.iloc[:sample_size].copy()
    else:
        dataframe = dataframe.copy()

    dataframe["text"] = dataframe["text"].fillna("").astype(str)
    dataframe["request_id"] = [
        f"subs-{idx:06d}" for idx in range(len(dataframe.index))
    ]

    logger.info(
        "Loaded {} rows from {} into in-memory dataframe.",
        len(dataframe.index),
        subs_path,
    )

    return dataframe


def materialize_subs_sample(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    subs_dir: Path | str = DEFAULT_SUBS_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    file_stem: str = DEFAULT_FILE_STEM,
) -> SubsSampleResult:
    """Persist a deterministic sample to ai_data as parquet and csv.
    
    Creates a unified file that can be reused with different limits.
    The file will contain up to `sample_size` messages, but the filename
    will not include the size, allowing reuse with --limit parameter.
    """

    dataframe = load_subs_messages(sample_size=sample_size, subs_dir=subs_dir)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = target_dir / f"{file_stem}.parquet"
    csv_path = target_dir / f"{file_stem}.csv"

    dataframe.to_parquet(parquet_path, index=False)
    dataframe.to_csv(csv_path, index=False)

    logger.info(
        "Stored subs sample ({} rows) at {} and {}",
        len(dataframe.index),
        parquet_path,
        csv_path,
    )

    return SubsSampleResult(
        dataframe=dataframe,
        parquet_path=parquet_path,
        csv_path=csv_path,
    )


def _read_parquet_slice(
    file_path: Path,
    rows_to_take: int,
    columns: Sequence[str],
    batch_size: int,
) -> pd.DataFrame:
    """Read up to `rows_to_take` rows from the parquet file."""

    if rows_to_take <= 0:
        return pd.DataFrame(columns=columns)

    if pq is None:
        dataframe = pd.read_parquet(file_path, columns=columns)
        return dataframe.head(rows_to_take)

    parquet_file = pq.ParquetFile(file_path)
    batches: list[pd.DataFrame] = []
    remaining = rows_to_take
    effective_batch_size = max(256, batch_size)

    for record_batch in parquet_file.iter_batches(
        batch_size=min(effective_batch_size, remaining),
        columns=columns,
    ):
        batch_df = record_batch.to_pandas()
        batches.append(batch_df)
        remaining -= len(batch_df.index)
        if remaining <= 0:
            break

    if not batches:
        return pd.DataFrame(columns=columns)

    dataframe = pd.concat(batches, ignore_index=True)
    return dataframe.iloc[:rows_to_take]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reproducible sample from ~/data/subs parquet files."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of messages to keep (default: 1000).",
    )
    parser.add_argument(
        "--subs-dir",
        default=str(DEFAULT_SUBS_DIR),
        help="Directory with source parquet files (default: ~/data/subs).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the sample will be stored (default: ai_data).",
    )
    parser.add_argument(
        "--file-stem",
        default=DEFAULT_FILE_STEM,
        help="Base file name for generated sample files.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = _parse_args()
    result = materialize_subs_sample(
        sample_size=args.limit,
        subs_dir=Path(args.subs_dir),
        output_dir=Path(args.output_dir),
        file_stem=args.file_stem,
    )

    print(
        f"Saved {len(result.dataframe.index)} rows to "
        f"{result.parquet_path} and {result.csv_path}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()


