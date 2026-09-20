"""
app/schemas.py

Pydantic request and response schemas for the RealTimeGuard inference API.

Design decisions:
- Enums for TransactionType, Prediction, and Decision give OpenAPI a
  closed vocabulary and reject invalid values before they reach the predictor.
- All numeric response fields are native Python ``float``/``int``.
  NumPy scalars are NOT allowed here — they cause JSON serialization errors.
- ``transaction_id`` is optional on the request (not all callers generate one)
  but always present in the response (generated as a UUID if not supplied).
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TransactionType(str, Enum):
    """Supported PaySim transaction types."""
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


class Prediction(str, Enum):
    """Binary fraud prediction label."""
    FRAUD = "FRAUD"
    LEGITIMATE = "LEGITIMATE"


class Decision(str, Enum):
    """
    Actionable decision derived from fraud probability thresholds.

    APPROVE  — fraud_probability < approve_threshold
    REVIEW   — approve_threshold ≤ prob ≤ review_threshold
    BLOCK    — fraud_probability > review_threshold
    """
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """
    Incoming transaction for fraud scoring.

    All monetary amounts and balances must be non-negative.
    ``type`` must be one of the five supported PaySim transaction types.
    """

    transaction_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional caller-supplied transaction identifier. "
            "A UUID is generated if not provided."
        ),
        examples=["txn_000001"],
    )
    step: int = Field(
        ...,
        ge=0,
        description="Simulation time step (1 step ≈ 1 hour).",
        examples=[120],
    )
    type: TransactionType = Field(
        ...,
        description="PaySim transaction type.",
        examples=["TRANSFER"],
    )
    amount: float = Field(
        ...,
        ge=0.0,
        description="Transaction amount (non-negative).",
        examples=[7500.0],
    )
    oldbalanceOrg: float = Field(
        ...,
        ge=0.0,
        description="Origin account balance before the transaction.",
        examples=[15000.0],
    )
    newbalanceOrig: float = Field(
        ...,
        ge=0.0,
        description="Origin account balance after the transaction.",
        examples=[7500.0],
    )
    oldbalanceDest: float = Field(
        ...,
        ge=0.0,
        description="Destination account balance before the transaction.",
        examples=[1000.0],
    )
    newbalanceDest: float = Field(
        ...,
        ge=0.0,
        description="Destination account balance after the transaction.",
        examples=[8500.0],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "transaction_id": "txn_000001",
                    "step": 120,
                    "type": "TRANSFER",
                    "amount": 7500.0,
                    "oldbalanceOrg": 15000.0,
                    "newbalanceOrig": 7500.0,
                    "oldbalanceDest": 1000.0,
                    "newbalanceDest": 8500.0,
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    """Structured fraud scoring result returned by POST /predict."""

    transaction_id: str = Field(
        description="Transaction identifier (caller-supplied or auto-generated UUID).",
        examples=["txn_000001"],
    )
    fraud_probability: float = Field(
        description="Model's estimated probability that this transaction is fraudulent.",
        examples=[0.94],
    )
    prediction: Prediction = Field(
        description="Binary fraud label.",
        examples=["FRAUD"],
    )
    decision: Decision = Field(
        description="Actionable decision based on configurable thresholds.",
        examples=["BLOCK"],
    )
    latency_ms: float = Field(
        description="End-to-end predictor latency in milliseconds.",
        examples=[8.4],
    )
    model_version: str = Field(
        description="Version of the loaded LightGBM model.",
        examples=["1.0.0"],
    )


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str = Field(
        description="'healthy' if all critical resources are loaded, 'degraded' otherwise.",
        examples=["healthy"],
    )
    model_loaded: bool = Field(description="True if the LightGBM model is loaded.")
    encoder_loaded: bool = Field(description="True if the LabelEncoder is loaded.")
    model_version: str = Field(
        description="Version string from model metadata.",
        examples=["1.0.0"],
    )
    service_uptime_seconds: float = Field(
        description="Seconds since the inference service started.",
        examples=[153.4],
    )


class MetricsResponse(BaseModel):
    """Response schema for GET /metrics."""

    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    fraud_predictions: int
    legitimate_predictions: int
    approve_decisions: int
    review_decisions: int
    block_decisions: int
