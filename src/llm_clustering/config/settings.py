"""Application settings and configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with flat structure."""

    # LLM Provider API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""

    # LLM Configuration
    llm_provider: str = "ollama"
    llm_model: str = "qwen3:30b"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096
    llm_prefer_dual_gpu: bool = True
    llm_allow_local_fallback: bool = True

    # Legacy LLM fields for compatibility
    default_llm_provider: str = "ollama"
    default_model: str = "qwen3:30b"
    default_temperature: float = 0.0
    default_max_tokens: int = 4096
    prefer_dual_gpu: bool = True
    allow_local_llm_fallback: bool = True

    # OpenRouter specific
    openrouter_model: str = "qwen/qwen-3-next"

    # Ollama specific (local)
    ollama_api_url: str = "http://localhost:11434/api"
    ollama_model: str = "qwen3:30b"

    # Triton specific (local)
    triton_api_url: str = "http://localhost:8000"
    triton_model: str = "qwen2_5_0_5b"

    # Clustering/Batch Configuration
    batch_size: int = 60
    max_clusters_per_batch: int = 8
    min_requests_per_cluster: int = 3
    rare_case_buffer_size: int = 20
    parallel_batch_size: int = 5
    clustering_similarity_threshold: float = 0.8

    # Legacy batch fields for compatibility
    clustering_batch_size: int = 60

    # Storage Configuration
    save_batches: bool = False
    save_slices: bool = False
    save_prompts: bool = True
    save_results: bool = True
    batches_dir: Path = Path("ai_data") / "batches"
    results_dir: Path = Path("ai_data") / "results"
    reports_dir: Path = Path("ai_data") / "reports"

    # Logging Configuration
    log_level: str = "INFO"
    log_file: Path = Path("ai_data/clustering.log")
    log_prompts: bool = True
    log_prompt_dir: Path = Path("ai_data/prompts")
    log_prompt_payloads: bool = False
    metrics_file: Path = Path("ai_data/metrics.csv")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
