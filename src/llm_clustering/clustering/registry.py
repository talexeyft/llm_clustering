"""Persistent Cluster Registry for MVP pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

from loguru import logger

from llm_clustering.config import Settings, get_settings


@dataclass(slots=True)
class ClusterRecord:
    """Single cluster entry stored in the registry."""

    cluster_id: str
    name: str
    summary: str
    criteria: str = ""
    sample_requests: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    updated_at: str | None = None
    count: int = 0
    status: str = "active"
    batch_id: str | None = None
    llm_reasoning: str | None = None
    revision: int = 1

    def touch(self, request_id: str | None = None) -> None:
        """Update stats when a request gets assigned."""
        self.count += 1
        self.updated_at = datetime.now(tz=timezone.utc).isoformat()

        if request_id and request_id not in self.sample_requests:
            if len(self.sample_requests) < 5:
                self.sample_requests.append(request_id)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClusterRecord":
        """Restore record from persisted payload."""
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialize record to dict."""
        return asdict(self)


class ClusterRegistry:
    """Simple JSON-backed registry that can be upgraded later."""

    def __init__(
        self,
        storage_path: Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        default_path = Path("ai_data") / "cluster_registry.json"
        self.storage_path = storage_path or default_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._clusters: dict[str, ClusterRecord] = {}
        self._load()

    def list_clusters(self, limit: int | None = None) -> list[ClusterRecord]:
        """Return clusters sorted by count desc."""
        with self._lock:
            records = sorted(
                self._clusters.values(),
                key=lambda item: item.count,
                reverse=True,
            )
        return records[:limit] if limit else records

    def get_cluster(self, cluster_id: str) -> ClusterRecord | None:
        """Fetch cluster by id."""
        with self._lock:
            return self._clusters.get(cluster_id)

    def upsert(self, record: ClusterRecord) -> ClusterRecord:
        """Create or update cluster entry."""
        with self._lock:
            existing = self._clusters.get(record.cluster_id)
            if existing:
                record.revision = existing.revision + 1
                if not record.sample_requests:
                    record.sample_requests = existing.sample_requests
                if record.count == 0:
                    record.count = existing.count
            self._clusters[record.cluster_id] = record
            self._save_locked()
            logger.info("Cluster %s saved (rev=%s)", record.cluster_id, record.revision)
            return record

    def bulk_upsert(self, records: Iterable[ClusterRecord]) -> list[ClusterRecord]:
        """Upsert multiple clusters within one lock."""
        with self._lock:
            updated: list[ClusterRecord] = []
            for record in records:
                existing = self._clusters.get(record.cluster_id)
                if existing:
                    record.revision = existing.revision + 1
                    if not record.sample_requests:
                        record.sample_requests = existing.sample_requests
                    if record.count == 0:
                        record.count = existing.count
                self._clusters[record.cluster_id] = record
                updated.append(record)
            self._save_locked()
        logger.info("Persisted %d clusters to registry.", len(updated))
        return updated

    def record_assignment(self, cluster_id: str, request_id: str) -> ClusterRecord:
        """Increment counters when Assignment Judge attaches request."""
        with self._lock:
            record = self._clusters.get(cluster_id)
            if not record:
                raise KeyError(f"Cluster '{cluster_id}' not found in registry.")
            record.touch(request_id=request_id)
            self._save_locked()
            return record

    def to_json(self) -> dict[str, Any]:
        """Return in-memory state as JSON serializable structure."""
        with self._lock:
            return {
                "version": 1,
                "clusters": [record.to_dict() for record in self._clusters.values()],
            }

    def _load(self) -> None:
        if not self.storage_path.exists():
            logger.info("Cluster registry file %s not found, starting fresh.", self.storage_path)
            return

        with self.storage_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        clusters = payload.get("clusters", [])
        for item in clusters:
            record = ClusterRecord.from_dict(item)
            self._clusters[record.cluster_id] = record

        logger.info("Loaded %d clusters from registry.", len(self._clusters))

    def _save_locked(self) -> None:
        """Persist current state to disk (expects caller to hold lock)."""
        payload = self.to_json()
        with self.storage_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

