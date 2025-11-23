"""Pydantic models for LLM response validation."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ClusterProposal(BaseModel):
    """Schema for a single cluster proposal from LLM."""

    cluster_id: str
    name: str
    summary: str
    criteria: str = ""
    sample_request_ids: list[str] = Field(default_factory=list)
    llm_reasoning: str = ""

    @field_validator("cluster_id", mode="before")
    @classmethod
    def sanitize_cluster_id(cls, v: str) -> str:
        """Sanitize cluster_id to snake_case format."""
        if not v:
            return v
        slug = re.sub(r"[^a-z0-9_]+", "_", str(v).strip().lower())
        slug = re.sub(r"_+", "_", slug).strip("_")
        return slug

    @field_validator("sample_request_ids", mode="before")
    @classmethod
    def ensure_list(cls, v: list[str] | None) -> list[str]:
        """Ensure sample_request_ids is a list."""
        if v is None:
            return []
        if not isinstance(v, list):
            return []
        return [str(item) for item in v]


class ProposerResponse(BaseModel):
    """Schema for full proposer response from LLM."""

    batch_id: str
    clusters: list[ClusterProposal] = Field(default_factory=list)
    skipped_request_ids: list[str] = Field(default_factory=list)

    @field_validator("skipped_request_ids", mode="before")
    @classmethod
    def ensure_list(cls, v: list[str] | None) -> list[str]:
        """Ensure skipped_request_ids is a list."""
        if v is None:
            return []
        if not isinstance(v, list):
            return []
        return [str(item) for item in v]


class SuggestedCluster(BaseModel):
    """Schema for a suggested new cluster from judge."""

    cluster_id: str = ""
    name: str = ""
    summary: str = ""
    criteria: str = ""
    sample_request_ids: list[str] = Field(default_factory=list)
    llm_rationale: str = ""

    @field_validator("sample_request_ids", mode="before")
    @classmethod
    def ensure_list(cls, v: list[str] | None) -> list[str]:
        """Ensure sample_request_ids is a list."""
        if v is None:
            return []
        if not isinstance(v, list):
            return []
        return [str(item) for item in v]


class JudgeResponse(BaseModel):
    """Schema for judge response from LLM."""

    request_id: str
    decision: Literal["assign", "new_cluster", "skip"]
    cluster_id: str = ""
    confidence_text: str = "low"
    llm_rationale: str = ""
    suggested_cluster: SuggestedCluster | None = None

    @field_validator("decision", mode="before")
    @classmethod
    def normalize_decision(cls, v: str) -> str:
        """Normalize decision to valid value."""
        decision = str(v or "skip").strip().lower()
        if decision not in {"assign", "new_cluster", "skip"}:
            return "skip"
        return decision

    @field_validator("suggested_cluster", mode="before")
    @classmethod
    def parse_suggested_cluster(cls, v: dict | None) -> SuggestedCluster | None:
        """Parse suggested_cluster from dict."""
        if not v or not isinstance(v, dict):
            return None
        # Check if it has required fields
        if not v.get("name") or not v.get("summary"):
            return None
        return SuggestedCluster(**v)

