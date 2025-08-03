"""Global configuration handled via Pydantic BaseSettings.

All hyper-parameters and paths are centralised here so that both training and
inference use the same source of truth.  Values can be overridden by env vars
(e.g. `AGENT_LAMBDA_HINT=0.4`).
"""
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields from .env
    )
    
    # Loss weights
    lambda_hint: float = Field(default=0.3, description="Lambda weight for hint loss")
    mu_gt: float = Field(default=0.1, description="Mu weight for ground truth")

    # Training defaults
    base_model: str = Field(default="gemma_base.gguf", description="Base model filename")
    tools: List[str] = Field(default_factory=lambda: ["browse", "click", "extract"])

    # Paths - Following PROJECT_STRUCTURE.md
    # Datasets should be stored in models/datasets/ and link to external SSD
    data_dir: Path = Field(default=Path("models/datasets"), description="Data directory")
    datasets_external: Path = Field(default=Path("/media/jerem/jeux&travail/datasets"), description="External datasets path")
    
    # Model storage and cache
    model_dir: Path = Field(default=Path("models"), description="Model directory")
    model_cache_dir: Path = Field(default=Path("/media/jerem/jeux&travail/ml_models"), description="External model cache on SSD")
    
    # Logs should be stored in models/logs/
    logs_dir: Path = Field(default=Path("models/logs"), description="Training and inference logs directory")
    
    # Results and checkpoints
    results_dir: Path = Field(default=Path("models/results"), description="Training results and checkpoints")
    
    # Scripts directory
    scripts_dir: Path = Field(default=Path("models/scripts"), description="Operational scripts directory")


settings = Settings()
