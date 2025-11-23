"""Utilities for persisting prompt/response payloads for audit."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from loguru import logger

from llm_clustering.config import Settings, get_settings


@dataclass(slots=True)
class PromptLogEntry:
    """Payload stored on disk for each prompt-response interaction."""

    prompt_name: str
    batch_id: str
    slice_id: str | None
    prompt: Mapping[str, Any]
    response: Any
    latency_ms: float | None = None
    cost_estimate: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


class PromptLogger:
    """Writes prompt/response JSON files to ai_data/prompts for auditing."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.log_dir = Path(self.settings.log_prompt_dir)
        if self.settings.save_prompts:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        entry: PromptLogEntry,
    ) -> Path | None:
        """Persist entry to disk if prompt logging is enabled."""
        if not self.settings.save_prompts:
            return None

        file_name = f"{entry.prompt_name}_{entry.batch_id}_{uuid4().hex[:6]}.json"
        file_path = self.log_dir / file_name
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(entry), handle, ensure_ascii=False, indent=2)

        logger.debug("Prompt log stored at %s", file_path)
        return file_path

