"""Application settings and configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BatchConfig(BaseModel):
    """Parameters that control how requests are processed inside a batch."""

    batch_size: int = Field(
        60,
        ge=1,
        description="Number of requests passed to the LLM in a single batch.",
    )
    max_clusters_per_batch: int = Field(
        8,
        ge=1,
        description="Upper bound on clusters proposed within one batch.",
    )
    min_requests_per_cluster: int = Field(
        3,
        ge=1,
        description="Minimum requests required for a cluster to be persisted.",
    )
    rare_case_buffer_size: int = Field(
        20,
        ge=1,
        description="Size of the buffer that stores requests without a cluster.",
    )


class LLMConfig(BaseModel):
    """LLM provider and model defaults."""

    provider: str = Field(
        "ollama",
        description="Provider name used by llm.factory (openrouter, ollama, triton, etc.).",
    )
    model: str = Field(
        "qwen3:30b",
        description="Default model identifier for the selected provider.",
    )
    temperature: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="Generation temperature passed to the provider.",
    )
    max_tokens: int = Field(
        4096,
        ge=256,
        description="Maximum number of new tokens generated for responses.",
    )
    prefer_dual_gpu: bool = Field(
        True,
        description="Hint for API/backends that can utilize two GPUs by default.",
    )
    allow_local_fallback: bool = Field(
        True,
        description="Allow the orchestrator to fall back to local providers (Ollama/Triton).",
    )


class LoggingConfig(BaseModel):
    """Log-level and prompt logging toggles."""

    level: str = Field("INFO", description="Application log level.")
    log_file: Path = Field(
        Path("ai_data") / "clustering.log",
        description="Path to the rolling clustering log file.",
    )
    log_prompts: bool = Field(
        True,
        description="Toggle prompt/response logging for auditability.",
    )
    prompt_log_dir: Path = Field(
        Path("ai_data") / "prompts",
        description="Directory where prompt/response JSON files are stored.",
    )
    metrics_file: Path = Field(
        Path("ai_data") / "metrics.csv",
        description="CSV file that stores execution metrics per batch.",
    )


class Settings(BaseSettings):
    """Application settings."""

    # LLM Provider API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""

    # LLM Configuration
    default_llm_provider: str = "ollama"
    default_model: str = "qwen3:30b"

    # OpenRouter specific
    openrouter_model: str = "qwen/qwen-3-next"

    # Ollama specific (local)
    ollama_api_url: str = "http://localhost:11434/api"
    ollama_model: str = "qwen3:30b"

    # Triton specific (local)
    triton_api_url: str = "http://localhost:8000"
    triton_model: str = "qwen2_5_0_5b"

    default_temperature: float = 0.0
    default_max_tokens: int = 4096
    prefer_dual_gpu: bool = True
    allow_local_llm_fallback: bool = True

    # Clustering Configuration
    clustering_batch_size: int = 60
    max_clusters_per_batch: int = 8
    min_requests_per_cluster: int = 3
    rare_case_buffer_size: int = 20
    clustering_similarity_threshold: float = 0.8

    # Storage
    batches_dir: Path = Path("ai_data") / "batches"
    results_dir: Path = Path("ai_data") / "results"
    reports_dir: Path = Path("ai_data") / "reports"

    # Logging
    log_level: str = "INFO"
    log_file: str = "ai_data/clustering.log"
    log_prompts: bool = True
    log_prompt_dir: str = "ai_data/prompts"
    log_prompt_payloads: bool = False
    metrics_file: str = "ai_data/metrics.csv"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def batch_config(self) -> BatchConfig:
        """Return normalized batch configuration."""
        return BatchConfig(
            batch_size=self.clustering_batch_size,
            max_clusters_per_batch=self.max_clusters_per_batch,
            min_requests_per_cluster=self.min_requests_per_cluster,
            rare_case_buffer_size=self.rare_case_buffer_size,
        )

    @property
    def llm_config(self) -> LLMConfig:
        """Return normalized LLM configuration."""
        return LLMConfig(
            provider=self.default_llm_provider,
            model=self.default_model or self.openrouter_model,
            temperature=self.default_temperature,
            max_tokens=self.default_max_tokens,
            prefer_dual_gpu=self.prefer_dual_gpu,
            allow_local_fallback=self.allow_local_llm_fallback,
        )

    @property
    def logging_config(self) -> LoggingConfig:
        """Return logging configuration."""
        return LoggingConfig(
            level=self.log_level,
            log_file=Path(self.log_file),
            log_prompts=self.log_prompts,
            prompt_log_dir=Path(self.log_prompt_dir),
            metrics_file=Path(self.metrics_file),
        )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

