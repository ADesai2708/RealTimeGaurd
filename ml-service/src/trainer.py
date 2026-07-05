"""
trainer.py

Reusable model training component for the RealTimeGuard
fraud detection pipeline.
"""

import time
from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_DIR, MODEL_VERSION
from src.preprocessing import DataPreprocessor
from src.utils import get_feature_list, save_json, setup_logger

logger = setup_logger(__name__)


class ModelTrainer:
    """
    Orchestrates model training, artifact saving, and
    preprocessing config persistence.

    Parameters
    ----------
    model : LGBMClassifier
        A configured (but not yet fitted) LightGBM classifier.
    preprocessor : DataPreprocessor
        A fitted DataPreprocessor instance (encoder already fit).
    model_dir : Path, optional
        Directory where artifacts are saved.
        Defaults to MODEL_DIR from config.
    """

    def __init__(
        self,
        model,
        preprocessor: DataPreprocessor,
        model_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.training_info: dict = {}

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> dict:
        """
        Fit the model and record training statistics.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature matrix for training.
        y_train : pd.Series
            Binary fraud labels (0 = normal, 1 = fraud).

        Returns
        -------
        dict
            Training info: duration_seconds, n_samples, n_features,
            n_fraud, n_normal, scale_pos_weight.
        """
        n_normal = int((y_train == 0).sum())
        n_fraud = int((y_train == 1).sum())
        scale_pos_weight = n_normal / n_fraud

        logger.info(
            "Starting training | samples=%d | fraud=%d | normal=%d | "
            "scale_pos_weight=%.4f",
            len(y_train),
            n_fraud,
            n_normal,
            scale_pos_weight,
        )

        # Inject scale_pos_weight before fitting
        self.model.set_params(scale_pos_weight=scale_pos_weight)

        start = time.perf_counter()
        self.model.fit(X_train, y_train)
        duration = time.perf_counter() - start

        logger.info("Training complete | duration=%.2fs", duration)

        self.training_info = {
            "duration_seconds": round(duration, 4),
            "n_samples": len(y_train),
            "n_features": X_train.shape[1],
            "n_fraud": n_fraud,
            "n_normal": n_normal,
            "scale_pos_weight": round(scale_pos_weight, 4),
            "model_version": MODEL_VERSION,
        }

        return self.training_info

    def save_artifacts(self, X_train: pd.DataFrame) -> None:
        """
        Persist all training artifacts:
        - Trained model (fraud_model.pkl)
        - Fitted label encoder (label_encoder.pkl)
        - Preprocessing configuration (preprocessing_config.json)

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix (used to extract feature list).
        """
        # 1. Save trained model
        model_path = self.model_dir / "fraud_model.pkl"
        joblib.dump(self.model, model_path)
        logger.info("Model saved → %s", model_path)

        # 2. Save fitted encoder
        encoder_path = self.model_dir / "label_encoder.pkl"
        self.preprocessor.save_encoder(encoder_path)
        logger.info("Encoder saved → %s", encoder_path)

        # 3. Save preprocessing config
        feature_list = get_feature_list(X_train)
        encoder_classes = (
            self.preprocessor.label_encoder.classes_.tolist()
            if hasattr(self.preprocessor.label_encoder, "classes_")
            else []
        )

        preprocessing_config = {
            "preprocessing_version": "1.0",
            "feature_list": feature_list,
            "encoder_classes": encoder_classes,
        }

        config_path = self.model_dir / "preprocessing_config.json"
        save_json(preprocessing_config, config_path)
        logger.info("Preprocessing config saved → %s", config_path)
