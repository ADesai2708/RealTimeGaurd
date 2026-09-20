"""
tests/test_predict.py

Tests for POST /predict.

Includes:
- Schema validation (missing fields, invalid types, negative amounts).
- Integration test using the actual saved model artifacts.
- Response schema verification.
- Decision threshold sanity checks.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """TestClient with full lifespan — model loads once for the module."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


VALID_TRANSFER_PAYLOAD = {
    "transaction_id": "txn_test_001",
    "step": 120,
    "type": "TRANSFER",
    "amount": 7500.0,
    "oldbalanceOrg": 15000.0,
    "newbalanceOrig": 7500.0,
    "oldbalanceDest": 1000.0,
    "newbalanceDest": 8500.0,
}

VALID_LEGITIMATE_PAYLOAD = {
    "step": 5,
    "type": "PAYMENT",
    "amount": 50.0,
    "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 4950.0,
    "oldbalanceDest": 200.0,
    "newbalanceDest": 250.0,
}


# ---------------------------------------------------------------------------
# Happy-path integration tests (uses real saved model)
# ---------------------------------------------------------------------------

class TestPredictHappyPath:
    def test_valid_transfer_returns_200(self, client):
        resp = client.post("/predict", json=VALID_TRANSFER_PAYLOAD)
        assert resp.status_code == 200, resp.text

    def test_response_has_all_required_fields(self, client):
        resp = client.post("/predict", json=VALID_TRANSFER_PAYLOAD)
        data = resp.json()
        required = {
            "transaction_id",
            "fraud_probability",
            "prediction",
            "decision",
            "latency_ms",
            "model_version",
        }
        assert required.issubset(data.keys()), f"Missing keys: {required - data.keys()}"

    def test_transaction_id_echoed(self, client):
        resp = client.post("/predict", json=VALID_TRANSFER_PAYLOAD)
        assert resp.json()["transaction_id"] == "txn_test_001"

    def test_transaction_id_generated_when_absent(self, client):
        payload = VALID_TRANSFER_PAYLOAD.copy()
        del payload["transaction_id"]
        data = client.post("/predict", json=payload).json()
        assert data["transaction_id"] != ""
        assert len(data["transaction_id"]) > 0

    def test_fraud_probability_is_float_between_0_and_1(self, client):
        data = client.post("/predict", json=VALID_TRANSFER_PAYLOAD).json()
        p = data["fraud_probability"]
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_prediction_is_valid_label(self, client):
        data = client.post("/predict", json=VALID_TRANSFER_PAYLOAD).json()
        assert data["prediction"] in ("FRAUD", "LEGITIMATE")

    def test_decision_is_valid_value(self, client):
        data = client.post("/predict", json=VALID_TRANSFER_PAYLOAD).json()
        assert data["decision"] in ("APPROVE", "REVIEW", "BLOCK")

    def test_latency_ms_is_positive(self, client):
        data = client.post("/predict", json=VALID_TRANSFER_PAYLOAD).json()
        assert data["latency_ms"] > 0

    def test_model_version_is_string(self, client):
        data = client.post("/predict", json=VALID_TRANSFER_PAYLOAD).json()
        assert isinstance(data["model_version"], str)
        assert data["model_version"] != ""

    def test_all_five_transaction_types_accepted(self, client):
        for tx_type in ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"):
            payload = VALID_LEGITIMATE_PAYLOAD.copy()
            payload["type"] = tx_type
            resp = client.post("/predict", json=payload)
            assert resp.status_code == 200, f"Type {tx_type} failed: {resp.text}"


# ---------------------------------------------------------------------------
# Input validation (must return 422 Unprocessable Entity)
# ---------------------------------------------------------------------------

class TestPredictValidation:
    def test_invalid_transaction_type_returns_422(self, client):
        payload = VALID_TRANSFER_PAYLOAD.copy()
        payload["type"] = "WIRE_TRANSFER"  # not a valid type
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_negative_amount_returns_422(self, client):
        payload = VALID_TRANSFER_PAYLOAD.copy()
        payload["amount"] = -100.0
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_negative_old_balance_org_returns_422(self, client):
        payload = VALID_TRANSFER_PAYLOAD.copy()
        payload["oldbalanceOrg"] = -1.0
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_negative_new_balance_orig_returns_422(self, client):
        payload = VALID_TRANSFER_PAYLOAD.copy()
        payload["newbalanceOrig"] = -0.01
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_missing_amount_returns_422(self, client):
        payload = VALID_TRANSFER_PAYLOAD.copy()
        del payload["amount"]
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_missing_step_returns_422(self, client):
        payload = VALID_TRANSFER_PAYLOAD.copy()
        del payload["step"]
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_missing_type_returns_422(self, client):
        payload = VALID_TRANSFER_PAYLOAD.copy()
        del payload["type"]
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_negative_step_returns_422(self, client):
        payload = VALID_TRANSFER_PAYLOAD.copy()
        payload["step"] = -1
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_empty_body_returns_422(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Decision threshold logic
# ---------------------------------------------------------------------------

class TestDecisionThresholds:
    """
    Verify that the decision engine applies thresholds correctly.

    We can't control the exact model output, but we can verify that
    the decision is internally consistent with the probability returned.
    """

    def _get_result(self, client, payload):
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        return resp.json()

    def test_decision_consistent_with_probability_approve(self, client):
        """If decision is APPROVE, probability must be < 0.40."""
        data = self._get_result(client, VALID_LEGITIMATE_PAYLOAD)
        if data["decision"] == "APPROVE":
            assert data["fraud_probability"] < 0.40

    def test_decision_consistent_with_probability_review(self, client):
        """If decision is REVIEW, probability must be in [0.40, 0.80]."""
        data = self._get_result(client, VALID_TRANSFER_PAYLOAD)
        if data["decision"] == "REVIEW":
            assert 0.40 <= data["fraud_probability"] <= 0.80

    def test_decision_consistent_with_probability_block(self, client):
        """If decision is BLOCK, probability must be > 0.80."""
        data = self._get_result(client, VALID_TRANSFER_PAYLOAD)
        if data["decision"] == "BLOCK":
            assert data["fraud_probability"] > 0.80

    def test_fraud_prediction_consistent_with_probability(self, client):
        """FRAUD prediction requires probability >= 0.5."""
        data = self._get_result(client, VALID_TRANSFER_PAYLOAD)
        if data["prediction"] == "FRAUD":
            assert data["fraud_probability"] >= 0.5
        else:
            assert data["fraud_probability"] < 0.5
