"""
app/main.py

FastAPI application entry point for the RealTimeGuard ML Inference Service.

Lifecycle:
    Startup  → configure logging → load model artifacts → initialise metrics
    Shutdown → log goodbye (artifacts are garbage-collected by Python)

Usage:
    python -m uvicorn app.main:app --reload     (development)
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000  (production)

Run from the ml-service/ directory so that relative artifact paths resolve.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.dependencies import load_resources
from app.logging_config import configure_logging
from app.metrics import InferenceMetrics
from app.routes import router
from app.settings import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup and shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Everything before ``yield`` runs at startup.
    Everything after ``yield`` runs at shutdown.

    Using lifespan instead of the deprecated @app.on_event("startup")
    decorator gives us structured startup/shutdown with proper exception
    propagation and avoids deprecation warnings in FastAPI 0.95+.
    """
    settings = get_settings()

    # 1. Configure logging first so all subsequent startup logs are formatted
    configure_logging(level=settings.log_level)

    # 2. Load model artifacts — fail fast if anything is missing
    resources = load_resources(settings)

    # 3. Initialise runtime metrics collector
    metrics = InferenceMetrics()

    # 4. Record service start time for uptime reporting in /health
    app.state.start_time = time.time()
    app.state.resources = resources
    app.state.metrics = metrics

    logger.info(
        "RealTimeGuard Inference Service started. Model v%s ready.",
        resources.model_version,
    )

    yield  # ← application runs here

    # Shutdown
    logger.info("RealTimeGuard Inference Service shutting down.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

settings = get_settings()

app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Register all API routes (POST /predict, GET /health, GET /metrics)
app.include_router(router)


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

@app.get(
    "/",
    summary="Service root",
    description="Identifies the RealTimeGuard ML Inference Service.",
    tags=["Meta"],
    include_in_schema=True,
)
async def root() -> JSONResponse:
    """
    Root endpoint — service identification.

    Returns a JSON object confirming the service name and API version.
    Useful for a quick sanity check that the server is reachable.
    """
    return JSONResponse(
        content={
            "service": "RealTimeGuard — ML Inference Service",
            "version": settings.api_version,
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
        }
    )
