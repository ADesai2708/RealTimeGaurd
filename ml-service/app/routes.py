"""
app/routes.py

API route handlers for the RealTimeGuard inference service.

Routes:
    POST /predict   — fraud scoring for a single transaction
    GET  /health    — readiness-style health check
    GET  /metrics   — runtime inference statistics

Design notes:
-------------
Liveness vs. Readiness:
    A pure liveness probe just checks "is the process alive?"  It can return
    200 even if the model is not loaded.  A readiness probe checks "is the
    service ready to serve traffic?" — it verifies that the model, encoder,
    and config are actually loaded.

    For this phase we implement a practical combined endpoint that acts as a
    readiness check: it reports the true state of loaded resources and returns
    "degraded" status if critical resources are absent.  This is suitable for
    Docker health-check usage and demo deployments.  In a full Kubernetes
    setup, you would split these into separate /livez and /readyz endpoints.

Error handling:
    - Pydantic validation errors → FastAPI automatically returns HTTP 422.
    - Predictor exceptions → caught here, logged, returned as HTTP 500.
    - Stack traces are NEVER sent to API clients.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.metrics import InferenceMetrics
from app.predictor import FeatureMismatchError, predictor
from app.schemas import (
    Decision,
    HealthResponse,
    MetricsResponse,
    Prediction,
    PredictionRequest,
    PredictionResponse,
)
from app.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score a transaction for fraud",
    description=(
        "Submit a single financial transaction for real-time fraud scoring. "
        "Returns a fraud probability, binary prediction, and an actionable "
        "decision (APPROVE / REVIEW / BLOCK)."
    ),
    tags=["Inference"],
)
async def predict(request_body: PredictionRequest, request: Request) -> PredictionResponse:
    """
    Run fraud inference on a single transaction.

    - **transaction_id**: optional; a UUID is generated if omitted.
    - **type**: must be one of CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER.
    - **amount / balances**: must be non-negative floats.
    """
    resources = request.app.state.resources
    metrics: InferenceMetrics = request.app.state.metrics

    try:
        result = predictor.predict(request_body, resources, settings)
    except FeatureMismatchError as exc:
        logger.error("Feature mismatch during prediction: %s", exc)
        metrics.record_failure()
        raise HTTPException(
            status_code=500,
            detail="Internal prediction error: feature mismatch. Check service logs.",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Unexpected prediction failure for txn=%s: %s",
            request_body.transaction_id or "unknown",
            exc,
        )
        metrics.record_failure()
        raise HTTPException(
            status_code=500,
            detail="Internal prediction error. Check service logs.",
        )

    # Update runtime metrics
    metrics.record_prediction(
        latency_ms=result.latency_ms,
        is_fraud=(result.prediction == Prediction.FRAUD),
        decision=result.decision.value,
        success=True,
    )

    return result


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Returns the readiness status of the inference service. "
        "Checks that the LightGBM model and LabelEncoder are loaded. "
        "Status is 'healthy' only when all critical resources are available."
    ),
    tags=["Operations"],
)
async def health(request: Request) -> HealthResponse:
    """
    Readiness-style health check.

    Returns ``status: healthy`` when model and encoder are loaded,
    ``status: degraded`` otherwise.  Suitable for Docker HEALTHCHECK usage.
    """
    resources = getattr(request.app.state, "resources", None)
    start_time: float = getattr(request.app.state, "start_time", time.time())

    model_loaded = resources is not None and hasattr(resources, "model")
    encoder_loaded = resources is not None and hasattr(resources, "encoder")
    model_version = resources.model_version if resources else "unknown"

    status = "healthy" if (model_loaded and encoder_loaded) else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        encoder_loaded=encoder_loaded,
        model_version=model_version,
        service_uptime_seconds=round(time.time() - start_time, 2),
    )


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Runtime inference metrics",
    description=(
        "Returns in-process prediction counters and latency percentiles. "
        "⚠ These metrics reflect only the current process. "
        "Aggregation across multiple workers or containers requires an "
        "external store (Redis, Prometheus) — planned for a later phase."
    ),
    tags=["Operations"],
)
async def metrics_endpoint(request: Request) -> MetricsResponse:
    """
    Return runtime inference statistics for this process.

    Latency percentiles are computed on-demand from a bounded buffer of
    the last 10,000 prediction latencies.
    """
    metrics: InferenceMetrics = request.app.state.metrics
    data = metrics.to_dict()
    return MetricsResponse(**data)
