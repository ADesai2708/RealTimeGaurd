"""
tests/test_trainer.py

Unit tests for src/trainer.py (ModelTrainer)

Uses a tiny synthetic dataset — no real PaySim data needed.
"""

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.model import build_model
from src.preprocessing import DataPreprocessor
from src.trainer import ModelTrainer


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_dataset():
    """Synthetic binary classification dataset (200 samples, 8 features)."""
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.9, 0.1],
        random_state=42,
    )
    feature_names = [
        "step", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "balanceChange", "hour",
    ]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="isFraud")
    return X_df, y_series


@pytest.fixture
def trained_trainer(tiny_dataset, tmp_path):
    """A ModelTrainer that has been trained on the tiny dataset."""
    X_df, y_series = tiny_dataset
    model = build_model(scale_pos_weight=1.0)
    preprocessor = DataPreprocessor()
    # Manually set a dummy encoder (not needed for numeric-only data)
    trainer = ModelTrainer(model=model, preprocessor=preprocessor, model_dir=tmp_path)
    trainer.train(X_df, y_series)
    return trainer, X_df


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

class TestModelTrainer:

    def test_train_returns_dict(self, tiny_dataset, tmp_path):
        X_df, y_series = tiny_dataset
        model = build_model(scale_pos_weight=1.0)
        preprocessor = DataPreprocessor()
        trainer = ModelTrainer(model=model, preprocessor=preprocessor, model_dir=tmp_path)
        info = trainer.train(X_df, y_series)
        assert isinstance(info, dict)

    def test_training_info_keys(self, tiny_dataset, tmp_path):
        X_df, y_series = tiny_dataset
        model = build_model(scale_pos_weight=1.0)
        preprocessor = DataPreprocessor()
        trainer = ModelTrainer(model=model, preprocessor=preprocessor, model_dir=tmp_path)
        info = trainer.train(X_df, y_series)
        expected_keys = {
            "duration_seconds", "n_samples", "n_features",
            "n_fraud", "n_normal", "scale_pos_weight", "model_version",
        }
        assert expected_keys.issubset(set(info.keys()))

    def test_training_info_values_are_positive(self, tiny_dataset, tmp_path):
        X_df, y_series = tiny_dataset
        model = build_model(scale_pos_weight=1.0)
        preprocessor = DataPreprocessor()
        trainer = ModelTrainer(model=model, preprocessor=preprocessor, model_dir=tmp_path)
        info = trainer.train(X_df, y_series)
        assert info["duration_seconds"] > 0
        assert info["n_samples"] == len(y_series)
        assert info["n_features"] == X_df.shape[1]
        assert info["n_fraud"] > 0
        assert info["n_normal"] > 0

    def test_scale_pos_weight_is_correct(self, tiny_dataset, tmp_path):
        X_df, y_series = tiny_dataset
        model = build_model(scale_pos_weight=1.0)
        preprocessor = DataPreprocessor()
        trainer = ModelTrainer(model=model, preprocessor=preprocessor, model_dir=tmp_path)
        info = trainer.train(X_df, y_series)
        expected = (y_series == 0).sum() / (y_series == 1).sum()
        assert abs(info["scale_pos_weight"] - round(expected, 4)) < 0.01

    def test_model_artifact_saved(self, trained_trainer):
        trainer, X_df = trained_trainer
        trainer.save_artifacts(X_df)
        assert (trainer.model_dir / "fraud_model.pkl").exists()

    def test_preprocessing_config_saved(self, trained_trainer):
        trainer, X_df = trained_trainer
        trainer.preprocessor.label_encoder.fit(["CASH_OUT", "TRANSFER"])
        trainer.save_artifacts(X_df)
        assert (trainer.model_dir / "preprocessing_config.json").exists()
