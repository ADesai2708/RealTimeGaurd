"""
tests/test_predictor.py

Unit tests for the Predictor inference engine.

These tests focus on:
- exact feature ordering
- inference consistency
- model resource reuse
- persisted preprocessing threshold
- prediction output validity
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from app.dependencies import load_resources
from app.predictor import Predictor
from app.schemas import PredictionRequest
from app.settings import get_settings


VALID_REQUEST = PredictionRequest(
    transaction_id="txn_predictor_test",
    step=120,
    type="TRANSFER",
    amount=7500.0,
    oldbalanceOrg=15000.0,
    newbalanceOrig=7500.0,
    oldbalanceDest=1000.0,
    newbalanceDest=8500.0,
)


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def resources(settings):
    return load_resources(settings)


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


class TestPredictor:

    def test_prediction_returns_valid_response(
        self,
        predictor,
        resources,
        settings,
    ):
        result = predictor.predict(
            VALID_REQUEST,
            resources,
            settings,
        )

        assert result.transaction_id == "txn_predictor_test"
        assert 0.0 <= result.fraud_probability <= 1.0
        assert result.prediction.value in {"FRAUD", "LEGITIMATE"}
        assert result.decision.value in {"APPROVE", "REVIEW", "BLOCK"}
        assert result.latency_ms > 0
        assert result.model_version == resources.model_version

    def test_feature_count_matches_training(
        self,
        resources,
    ):
        assert len(resources.feature_list) == 14
        assert len(resources.feature_list) == resources.metadata["n_features"]

    def test_persisted_threshold_is_loaded(
        self,
        resources,
    ):
        assert resources.large_transaction_threshold > 0
        assert (
            "large_transaction_threshold"
            in resources.preprocessing_config
        )

    def test_same_input_produces_consistent_probability(
        self,
        predictor,
        resources,
        settings,
    ):
        first = predictor.predict(
            VALID_REQUEST,
            resources,
            settings,
        )

        second = predictor.predict(
            VALID_REQUEST,
            resources,
            settings,
        )

        assert first.fraud_probability == second.fraud_probability
        assert first.prediction == second.prediction
        assert first.decision == second.decision

    def test_predictor_does_not_replace_loaded_model(
        self,
        predictor,
        resources,
        settings,
    ):
        original_model = resources.model

        predictor.predict(
            VALID_REQUEST,
            resources,
            settings,
        )

        assert resources.model is original_model

    def test_model_is_loaded_once_per_resource_instance(
        self,
        settings,
    ):
        """
        load_resources() creates one model object for one resource instance.

        Predictor.predict() must reuse that object rather than loading
        the model from disk again.
        """
        resources = load_resources(settings)

        model_before = resources.model
        predictor = Predictor()

        predictor.predict(
            VALID_REQUEST,
            resources,
            settings,
        )

        predictor.predict(
            VALID_REQUEST,
            resources,
            settings,
        )

        assert resources.model is model_before