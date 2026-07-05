"""
tests/test_evaluator.py

Unit tests for src/evaluator.py (ModelEvaluator)

Trains a tiny LightGBM model on synthetic data, then exercises
the full evaluator API — metrics, report generation, file output.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.evaluator import ModelEvaluator
from src.model import build_model


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fitted_evaluator(tmp_path_factory):
    """
    Fit a tiny LightGBM model and return a ModelEvaluator
    ready for testing.
    """
    tmp_path = tmp_path_factory.mktemp("reports")

    X, y = make_classification(
        n_samples=500,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.85, 0.15],
        random_state=42,
    )
    feature_names = [
        "step", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "balanceChange", "hour",
    ]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_s = pd.Series(y, name="isFraud")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_s, test_size=0.2, random_state=42, stratify=y_s
    )

    model = build_model(scale_pos_weight=5.0)
    model.fit(X_train, y_train)

    evaluator = ModelEvaluator(
        model=model,
        X_test=X_test,
        y_test=y_test,
        report_dir=tmp_path,
    )
    return evaluator


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

class TestModelEvaluator:

    def test_evaluate_returns_dict(self, fitted_evaluator):
        metrics = fitted_evaluator.evaluate()
        assert isinstance(metrics, dict)

    def test_metrics_keys_present(self, fitted_evaluator):
        metrics = fitted_evaluator.evaluate()
        expected_keys = {"accuracy", "precision", "recall", "f1_score",
                         "roc_auc", "confusion_matrix"}
        assert expected_keys.issubset(set(metrics.keys()))

    def test_confusion_matrix_keys(self, fitted_evaluator):
        metrics = fitted_evaluator.evaluate()
        cm = metrics["confusion_matrix"]
        assert set(cm.keys()) == {"TN", "FP", "FN", "TP"}

    def test_metrics_are_in_range(self, fitted_evaluator):
        metrics = fitted_evaluator.evaluate()
        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of [0,1] range"

    def test_confusion_matrix_values_are_non_negative(self, fitted_evaluator):
        metrics = fitted_evaluator.evaluate()
        cm = metrics["confusion_matrix"]
        for k, v in cm.items():
            assert v >= 0, f"{k} is negative"

    def test_generate_reports_raises_if_not_evaluated(self, tmp_path):
        """generate_reports() must raise before evaluate() is called."""
        X, y = make_classification(n_samples=50, n_features=8, random_state=0)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
        y_s = pd.Series(y)
        model = build_model(scale_pos_weight=1.0)
        evaluator = ModelEvaluator(model=model, X_test=X_df, y_test=y_s,
                                   report_dir=tmp_path)
        with pytest.raises(RuntimeError, match="evaluate()"):
            evaluator.generate_reports()

    def test_confusion_matrix_png_created(self, fitted_evaluator):
        fitted_evaluator.generate_reports()
        assert (fitted_evaluator.report_dir / "confusion_matrix.png").exists()

    def test_roc_curve_png_created(self, fitted_evaluator):
        assert (fitted_evaluator.report_dir / "roc_curve.png").exists()

    def test_pr_curve_png_created(self, fitted_evaluator):
        assert (fitted_evaluator.report_dir / "pr_curve.png").exists()

    def test_feature_importance_png_created(self, fitted_evaluator):
        assert (fitted_evaluator.report_dir / "feature_importance.png").exists()

    def test_metrics_json_created(self, fitted_evaluator):
        metrics_path = fitted_evaluator.report_dir / "metrics.json"
        assert metrics_path.exists()

    def test_metrics_json_is_valid(self, fitted_evaluator):
        metrics_path = fitted_evaluator.report_dir / "metrics.json"
        with open(metrics_path) as f:
            data = json.load(f)
        assert "accuracy" in data
        assert "roc_auc" in data
