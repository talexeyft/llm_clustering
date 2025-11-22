"""Helpers for writing batch metrics to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from llm_clustering.config import Settings, get_settings


class MetricsTracker:
    """Appends batch-level metrics into a CSV file."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.metrics_path = Path(self.settings.metrics_file)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict[str, Any]) -> Path:
        """Append a single metrics row."""
        dataframe = pd.DataFrame([row])
        header = not self.metrics_path.exists()
        dataframe.to_csv(
            self.metrics_path,
            mode="a",
            header=header,
            index=False,
        )
        return self.metrics_path

