"""
tests/test_health.py

Tests for GET / and GET /health endpoints.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """TestClient with full lifespan — model loads once for the module."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# GET / — root endpoint
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_contains_service_name(self, client):
        data = resp = client.get("/")
        data = resp.json()
        assert "RealTimeGuard" in data["service"]

    def test_root_contains_version(self, client):
        data = client.get("/").json()
        assert "version" in data

    def test_root_contains_docs_link(self, client):
        data = client.get("/").json()
        assert data["docs"] == "/docs"

    def test_root_contains_predict_link(self, client):
        data = client.get("/").json()
        assert data["predict"] == "/predict"


# ---------------------------------------------------------------------------
# GET /health — readiness check
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_is_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_model_is_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_encoder_is_loaded(self, client):
        data = client.get("/health").json()
        assert data["encoder_loaded"] is True

    def test_model_version_present(self, client):
        data = client.get("/health").json()
        assert "model_version" in data
        assert data["model_version"] != ""

    def test_uptime_is_positive(self, client):
        data = client.get("/health").json()
        assert data["service_uptime_seconds"] >= 0

    def test_health_response_schema(self, client):
        data = client.get("/health").json()
        required_keys = {
            "status",
            "model_loaded",
            "encoder_loaded",
            "model_version",
            "service_uptime_seconds",
        }
        assert required_keys.issubset(data.keys())
