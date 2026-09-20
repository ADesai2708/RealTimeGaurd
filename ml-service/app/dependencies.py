"""
app/dependencies.py

Resource loading for the RealTimeGuard inference service.

Why load once?
--------------
Loading fraud_model.pkl with joblib takes ~50–200 ms depending on hardware.
The LabelEncoder and JSON configs add another ~5–20 ms.  If these were
loaded inside ``/predict``, every request would pay this overhead before
seeing a single inference operation.  Phase 1 benchmarks showed the model's
``predict_proba`` itself takes ~1 ms.  Paying 100x that cost per request in
I/O would be wasteful and unpredictable under load.

Instead, ``load_resources()`` is called once during the FastAPI lifespan
startup hook.  The loaded objects are stored on ``app.state`` and injected
into route handlers through FastAPI's dependency injection system.

Fail-fast behavior:
-------------------
If any required artifact is missing or malformed, ``load_resources()``
raises a ``RuntimeError`` with a descriptive message.  This is intentional:
a service that can't load its model should never accept traffic.  A bad
startup error is infinitely better than silently serving wrong predictions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.preprocessing import LabelEncoder

from app.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class ModelResources:
    """
    Container for all inference-time artifacts.

    Loaded once at startup and stored on ``app.state.resources``.
    Passed to the predictor on every request.

    Attributes
    ----------
    model : LGBMClassifier
        Fitted LightGBM model with ``predict_proba``.
    encoder : LabelEncoder
        Fitted encoder for the transaction ``type`` column.
    metadata : dict
        Contents of ``model_metadata.json`` (version, feature list, etc.).
    preprocessing_config : dict
        Contents of ``preprocessing_config.json`` (feature list,
        encoder classes, large_transaction_threshold).
    model_version : str
        Convenience accessor for the model version string.
    feature_list : list[str]
        Ordered list of 14 feature names expected by the model.
    large_transaction_threshold : float
        The 95th-percentile ``amount`` threshold computed during training.
        Used by the feature engineering step during inference to avoid
        training-serving skew.
    """

    model: Any
    encoder: LabelEncoder
    metadata: dict
    preprocessing_config: dict
    model_version: str
    feature_list: list[str]
    large_transaction_threshold: float


def load_resources(settings: Settings) -> ModelResources:
    """
    Load all model artifacts from disk and return a validated
    ``ModelResources`` instance.

    Parameters
    ----------
    settings : Settings
        Application settings providing artifact paths.

    Returns
    -------
    ModelResources

    Raises
    ------
    RuntimeError
        If any artifact is missing, unreadable, or fails validation.
    """
    logger.info("=" * 60)
    logger.info("RealTimeGuard Inference Service — loading artifacts …")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Validate artifact paths before loading
    # ------------------------------------------------------------------
    required_paths = {
        "Model": settings.model_path,
        "LabelEncoder": settings.encoder_path,
        "Metadata": settings.metadata_path,
        "PreprocessingConfig": settings.preprocessing_config_path,
    }
    for name, path in required_paths.items():
        if not Path(path).exists():
            raise RuntimeError(
                f"Required artifact not found: {name} → {path}\n"
                "Ensure Phase 1 training has been completed and all artifacts "
                "are present in the models/ directory."
            )

    # ------------------------------------------------------------------
    # 2. Load and validate model
    # ------------------------------------------------------------------
    logger.info("Loading model from: %s", settings.model_path)
    model = joblib.load(settings.model_path)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            f"Loaded object from {settings.model_path} does not expose "
            "'predict_proba'. Expected a fitted LGBMClassifier."
        )
    logger.info("Model loaded successfully. Type: %s", type(model).__name__)

    # ------------------------------------------------------------------
    # 3. Load and validate encoder
    # ------------------------------------------------------------------
    logger.info("Loading encoder from: %s", settings.encoder_path)
    encoder: LabelEncoder = joblib.load(settings.encoder_path)

    if not hasattr(encoder, "classes_"):
        raise RuntimeError(
            f"Loaded encoder from {settings.encoder_path} does not have "
            "'classes_' attribute. The encoder may not be fitted."
        )
    logger.info(
        "Encoder loaded. Known classes: %s", encoder.classes_.tolist()
    )

    # ------------------------------------------------------------------
    # 4. Load and validate model metadata
    # ------------------------------------------------------------------
    logger.info("Loading metadata from: %s", settings.metadata_path)
    with open(settings.metadata_path) as f:
        metadata: dict = json.load(f)

    required_metadata_keys = {"model_version", "feature_list", "n_features"}
    missing = required_metadata_keys - metadata.keys()
    if missing:
        raise RuntimeError(
            f"model_metadata.json is missing required keys: {missing}"
        )

    model_version: str = metadata["model_version"]
    feature_list: list[str] = metadata["feature_list"]
    n_features: int = metadata["n_features"]

    logger.info("Model version: %s", model_version)
    logger.info("Expected features (%d): %s", n_features, feature_list)

    # ------------------------------------------------------------------
    # 5. Load and validate preprocessing config
    # ------------------------------------------------------------------
    logger.info(
        "Loading preprocessing config from: %s",
        settings.preprocessing_config_path,
    )
    with open(settings.preprocessing_config_path) as f:
        preprocessing_config: dict = json.load(f)

    # Validate large_transaction_threshold is present.
    # Its absence means the config was saved by the old trainer.py before
    # the training-serving skew fix was applied.
    if "large_transaction_threshold" not in preprocessing_config:
        raise RuntimeError(
            "preprocessing_config.json is missing 'large_transaction_threshold'.\n"
            "This key is required to prevent training-serving skew in the "
            "'isLargeTransaction' feature.\n"
            "Fix options:\n"
            "  1. Re-run the training pipeline (python -m src.train) which now "
            "automatically persists this value.\n"
            "  2. Run the compute_threshold.py helper script to patch the "
            "existing config without retraining."
        )

    large_transaction_threshold = float(
        preprocessing_config["large_transaction_threshold"]
    )
    logger.info(
        "large_transaction_threshold loaded: %.4f",
        large_transaction_threshold,
    )

    logger.info("=" * 60)
    logger.info(
        "All artifacts loaded. Model v%s ready for inference.", model_version
    )
    logger.info("=" * 60)

    return ModelResources(
        model=model,
        encoder=encoder,
        metadata=metadata,
        preprocessing_config=preprocessing_config,
        model_version=model_version,
        feature_list=feature_list,
        large_transaction_threshold=large_transaction_threshold,
    )
