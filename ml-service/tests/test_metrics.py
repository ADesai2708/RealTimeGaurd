"""
tests/test_metrics.py

Tests for the InferenceMetrics runtime metrics collector.
"""

from __future__ import annotations

from app.metrics import InferenceMetrics


class TestInferenceMetrics:

    def test_initial_metrics_are_zero(self):
        metrics = InferenceMetrics()

        data = metrics.to_dict()

        assert data["total_predictions"] == 0
        assert data["successful_predictions"] == 0
        assert data["failed_predictions"] == 0

        assert data["fraud_predictions"] == 0
        assert data["legitimate_predictions"] == 0

        assert data["approve_decisions"] == 0
        assert data["review_decisions"] == 0
        assert data["block_decisions"] == 0

        assert data["average_latency_ms"] == 0.0
        assert data["p50_latency_ms"] == 0.0
        assert data["p95_latency_ms"] == 0.0
        assert data["p99_latency_ms"] == 0.0

    def test_successful_prediction_is_recorded(self):
        metrics = InferenceMetrics()

        metrics.record_prediction(
            latency_ms=10.0,
            is_fraud=False,
            decision="APPROVE",
            success=True,
        )

        data = metrics.to_dict()

        assert data["total_predictions"] == 1
        assert data["successful_predictions"] == 1
        assert data["failed_predictions"] == 0

        assert data["legitimate_predictions"] == 1
        assert data["fraud_predictions"] == 0

        assert data["approve_decisions"] == 1

    def test_fraud_prediction_is_recorded(self):
        metrics = InferenceMetrics()

        metrics.record_prediction(
            latency_ms=20.0,
            is_fraud=True,
            decision="BLOCK",
            success=True,
        )

        data = metrics.to_dict()

        assert data["total_predictions"] == 1
        assert data["successful_predictions"] == 1
        assert data["fraud_predictions"] == 1
        assert data["legitimate_predictions"] == 0
        assert data["block_decisions"] == 1

    def test_failed_prediction_is_recorded(self):
        metrics = InferenceMetrics()

        metrics.record_failure()

        data = metrics.to_dict()

        assert data["total_predictions"] == 1
        assert data["successful_predictions"] == 0
        assert data["failed_predictions"] == 1

    def test_latency_statistics_are_calculated(self):
        metrics = InferenceMetrics()

        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]

        for latency in latencies:
            metrics.record_prediction(
                latency_ms=latency,
                is_fraud=False,
                decision="APPROVE",
                success=True,
            )

        data = metrics.to_dict()

        assert data["total_predictions"] == 5
        assert data["successful_predictions"] == 5

        assert data["average_latency_ms"] == 30.0
        assert data["p50_latency_ms"] == 30.0

        assert data["p95_latency_ms"] > 0
        assert data["p99_latency_ms"] > 0

    def test_all_decisions_are_counted(self):
        metrics = InferenceMetrics()

        metrics.record_prediction(
            latency_ms=5.0,
            is_fraud=False,
            decision="APPROVE",
        )

        metrics.record_prediction(
            latency_ms=10.0,
            is_fraud=True,
            decision="REVIEW",
        )

        metrics.record_prediction(
            latency_ms=20.0,
            is_fraud=True,
            decision="BLOCK",
        )

        data = metrics.to_dict()

        assert data["approve_decisions"] == 1
        assert data["review_decisions"] == 1
        assert data["block_decisions"] == 1

        assert data["fraud_predictions"] == 2
        assert data["legitimate_predictions"] == 1

    def test_metrics_invariants(self):
        metrics = InferenceMetrics()

        metrics.record_prediction(
            latency_ms=5.0,
            is_fraud=False,
            decision="APPROVE",
        )

        metrics.record_prediction(
            latency_ms=10.0,
            is_fraud=True,
            decision="BLOCK",
        )

        metrics.record_failure()

        data = metrics.to_dict()

        assert (
            data["total_predictions"]
            == data["successful_predictions"] + data["failed_predictions"]
        )

        assert (
            data["successful_predictions"]
            == data["fraud_predictions"] + data["legitimate_predictions"]
        )