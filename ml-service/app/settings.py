"""
app/settings.py

Centralized inference service settings for RealTimeGuard.

All tuneable values live here. Paths are relative to the ml-service/
directory so there are no hardcoded absolute paths.

Environment variable overrides (prefix: RTGUARD_):

    RTGUARD_MODEL_PATH=models/fraud_model.pkl
    RTGUARD_APPROVE_THRESHOLD=0.40
    RTGUARD_REVIEW_THRESHOLD=0.80
    RTGUARD_LOG_LEVEL=INFO

Usage:
    from app.settings import get_settings
    settings = get_settings()
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Base directory — resolves to ml-service/ regardless of where the process
# is started from, as long as this file lives at ml-service/app/settings.py.
# ---------------------------------------------------------------------------
_ML_SERVICE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Inference service settings.

    Values can be overridden via environment variables with the prefix
    ``RTGUARD_``.  For example, setting ``RTGUARD_LOG_LEVEL=DEBUG``
    overrides ``log_level``.
    """

    model_config = SettingsConfigDict(
        env_prefix="RTGUARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # API metadata
    # ------------------------------------------------------------------
    api_title: str = "RealTimeGuard — ML Inference Service"
    api_description: str = (
        "Real-time transaction fraud detection using LightGBM. "
        "Part of the RealTimeGuard fraud detection platform."
    )
    api_version: str = "1.0.0"

    # ------------------------------------------------------------------
    # Artifact paths  (relative to ml-service/)
    # ------------------------------------------------------------------
    model_path: Path = _ML_SERVICE_DIR / "models" / "fraud_model.pkl"
    encoder_path: Path = _ML_SERVICE_DIR / "models" / "label_encoder.pkl"
    metadata_path: Path = _ML_SERVICE_DIR / "models" / "model_metadata.json"
    preprocessing_config_path: Path = (
        _ML_SERVICE_DIR / "models" / "preprocessing_config.json"
    )

    # ------------------------------------------------------------------
    # Decision thresholds
    #
    # Decision policy:
    #   fraud_probability < APPROVE_THRESHOLD          → APPROVE
    #   APPROVE_THRESHOLD ≤ prob ≤ REVIEW_THRESHOLD    → REVIEW
    #   fraud_probability > REVIEW_THRESHOLD           → BLOCK
    # ------------------------------------------------------------------
    approve_threshold: float = 0.40
    review_threshold: float = 0.80

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got: {v!r}")
        return v_upper

    @model_validator(mode="after")
    def validate_thresholds(self) -> "Settings":
        """Ensure APPROVE_THRESHOLD < REVIEW_THRESHOLD."""
        if self.approve_threshold >= self.review_threshold:
            raise ValueError(
                f"approve_threshold ({self.approve_threshold}) must be strictly less "
                f"than review_threshold ({self.review_threshold})"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Using lru_cache(maxsize=1) ensures settings are read from environment
    variables exactly once and reused across all requests.  Call
    ``get_settings.cache_clear()`` in tests that need to override settings.
    """
    return Settings()
