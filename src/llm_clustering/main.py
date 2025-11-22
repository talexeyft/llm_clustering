"""CLI entry point for running the MVP clustering pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from llm_clustering.pipeline import PipelineRunner
from llm_clustering.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM clustering pipeline on a dataset.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV or Parquet file containing customer requests.",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "csv", "parquet"],
        default="auto",
        help="Input file format. Defaults to auto-detect by extension.",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Optional batch identifier. If omitted it will be generated automatically.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column that stores the request text (default: text).",
    )
    return parser.parse_args()


def load_dataframe(path: Path, file_format: str) -> pd.DataFrame:
    if file_format == "auto":
        suffix = path.suffix.lower()
        if suffix == ".csv":
            file_format = "csv"
        elif suffix in {".parquet", ".pq"}:
            file_format = "parquet"
        else:
            raise ValueError(f"Cannot auto-detect format for {path}. Use --format.")

    if file_format == "csv":
        return pd.read_csv(path)
    if file_format == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported format: {file_format}")


def main() -> None:
    args = parse_args()
    setup_logger()

    input_path = Path(args.input)
    dataframe = load_dataframe(input_path, args.format)

    runner = PipelineRunner(text_column=args.text_column)
    result = runner.run(dataframe, batch_id=args.batch_id)

    total_requests = len(result.assignments_df.index)
    assigned = int(result.assignments_df["cluster_id"].notna().sum()) if total_requests else 0
    coverage = (assigned / total_requests) * 100 if total_requests else 0.0

    print(f"[LLM Clustering] Batch {result.batch_id} processed.")
    print(f"Requests: {total_requests}, assigned: {assigned} ({coverage:.1f}% coverage)")
    print(f"Assignments stored at: {result.assignments_path}")
    if result.cohesion_report:
        print(f"Cohesion check saved to: {result.cohesion_report}")
    else:
        print("Cohesion check skipped (not enough assignments).")
    print(f"Metrics appended to: {result.metrics_path}")
    print(f"Prepared snapshot: {result.prepared_snapshot.csv}")


if __name__ == "__main__":
    main()

