"""
tests/test_model.py

Unit tests for src/model.py
"""

import pytest
# pyrefly: ignore [missing-import]
from lightgbm import LGBMClassifier

from src.config import MODEL_PARAMS
from src.model import build_model


class TestBuildModel:

    def test_returns_lgbm_classifier(self):
        model = build_model(scale_pos_weight=50.0)
        assert isinstance(model, LGBMClassifier)

    def test_scale_pos_weight_is_set(self):
        model = build_model(scale_pos_weight=61.0)
        assert model.get_params()["scale_pos_weight"] == 61.0

    def test_objective_is_binary(self):
        model = build_model(scale_pos_weight=1.0)
        assert model.get_params()["objective"] == "binary"

    def test_n_estimators_matches_config(self):
        model = build_model(scale_pos_weight=1.0)
        assert model.get_params()["n_estimators"] == MODEL_PARAMS["n_estimators"]

    def test_learning_rate_matches_config(self):
        model = build_model(scale_pos_weight=1.0)
        assert model.get_params()["learning_rate"] == MODEL_PARAMS["learning_rate"]

    def test_max_depth_matches_config(self):
        model = build_model(scale_pos_weight=1.0)
        assert model.get_params()["max_depth"] == MODEL_PARAMS["max_depth"]

    def test_random_state_matches_config(self):
        model = build_model(scale_pos_weight=1.0)
        assert model.get_params()["random_state"] == MODEL_PARAMS["random_state"]

    def test_different_weights_create_distinct_models(self):
        m1 = build_model(scale_pos_weight=10.0)
        m2 = build_model(scale_pos_weight=50.0)
        assert m1.get_params()["scale_pos_weight"] != m2.get_params()["scale_pos_weight"]
