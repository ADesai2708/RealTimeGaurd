"""
app/predictor.py

Prediction engine for the RealTimeGuard inference service.

This module is the bridge between an HTTP request and the LightGBM model.
It is responsible for:

  1. Converting a validated PredictionRequest into a raw feature DataFrame.
  2. Applying the SAME feature engineering used during training
     (using the persisted large_transaction_threshold to avoid skew).
  3. Encoding the transaction type using the saved LabelEncoder (transform-only).
  4. Reordering features to the exact training order.
  5. Validating that the final feature count matches the model's expectation.
  6. Calling model.predict_proba and extracting the positive-class probability.
  7. Applying decision thresholds.
  8. Measuring per-call latency with time.perf_counter.
  9. Returning a PredictionResponse with all Python-native types.

What this module NEVER does:
  - Never calls fit() or fit_transform().
  - Never computes statistics (mean, quantile, std) from the request data.
  - Never loads model artifacts from disk (loaded once at startup).
  - Never silently reorder or drop features without validation.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from app.schemas import Decision, Prediction, PredictionRequest, PredictionResponse
from app.settings import Settings

if TYPE_CHECKING:
    from app.dependencies import ModelResources

logger = logging.getLogger(__name__)


class FeatureMismatchError(Exception):
    """
    Raised when the prepared feature DataFrame does not match what
    the model expects (wrong count, wrong names, or wrong order).
    """


class UnknownTransactionTypeError(Exception):
    """
    Raised when the transaction type is not in the encoder's known classes.
    Separate from Pydantic validation because the encoder check happens
    after schema validation.
    """


class Predictor:
    """
    Stateless prediction engine.

    All state (model, encoder, config) is injected at call time via
    ``ModelResources``.  The class itself holds no mutable state, so a
    single instance can safely serve concurrent requests.
    """

    def predict(
        self,
        request: PredictionRequest,
        resources: "ModelResources",
        settings: Settings,
    ) -> PredictionResponse:
        """
        Run end-to-end fraud prediction for a single transaction.

        Parameters
        ----------
        request : PredictionRequest
            Validated transaction data from the API layer.
        resources : ModelResources
            Loaded model artifacts (model, encoder, config).
        settings : Settings
            Application settings (decision thresholds).

        Returns
        -------
        PredictionResponse
            Structured prediction result with Python-native types.

        Raises
        ------
        FeatureMismatchError
            If the prepared feature set does not match training expectations.
        UnknownTransactionTypeError
            If the transaction type is not in the encoder's known classes.
        Exception
            Any unexpected error propagates up to the route handler which
            converts it to HTTP 500.
        """
        start = time.perf_counter()

        # ------------------------------------------------------------------
        # 1. Resolve transaction ID
        # ------------------------------------------------------------------
        transaction_id = request.transaction_id or str(uuid.uuid4())

        # ------------------------------------------------------------------
        # 2. Build raw feature dict
        #    Use the exact column names the training pipeline expected.
        #    The type is kept as a string here; encoding happens in step 4.
        # ------------------------------------------------------------------
        raw = {
            "step": request.step,
            "type": request.type.value,      # enum → string ("TRANSFER" etc.)
            "amount": request.amount,
            "oldbalanceOrg": request.oldbalanceOrg,
            "newbalanceOrig": request.newbalanceOrig,
            "oldbalanceDest": request.oldbalanceDest,
            "newbalanceDest": request.newbalanceDest,
        }

        # ------------------------------------------------------------------
        # 3. Create one-row DataFrame
        # ------------------------------------------------------------------
        df = pd.DataFrame([raw])

        # ------------------------------------------------------------------
        # 4. Feature engineering
        #
        #    Pass the persisted large_transaction_threshold so that
        #    isLargeTransaction is computed consistently with training.
        #    Without this, quantile(0.95) on a 1-row DataFrame equals the
        #    row's own amount → the flag is always 0 (training-serving skew).
        # ------------------------------------------------------------------
        from src.feature_engineering import add_engineered_features

        df = add_engineered_features(
            df,
            large_transaction_threshold=resources.large_transaction_threshold,
        )

        # ------------------------------------------------------------------
        # 5. Encode transaction type (transform-only — NEVER fit)
        #
        #    Validate the type is known before calling transform so we can
        #    surface a clear error instead of a cryptic sklearn exception.
        # ------------------------------------------------------------------
        type_value = raw["type"]
        if type_value not in resources.encoder.classes_:
            raise UnknownTransactionTypeError(
                f"Transaction type '{type_value}' is not in the encoder's "
                f"known classes: {resources.encoder.classes_.tolist()}"
            )

        df["type"] = resources.encoder.transform(df["type"])

        # ------------------------------------------------------------------
        # 6. Select and reorder features to the exact training order
        #
        #    This is the most dangerous step to get wrong:
        #    - Extra columns (e.g. isFraud) must be dropped.
        #    - Missing columns must raise loudly — not be silently filled.
        #    - Column ORDER must match training exactly.
        #      LightGBM stores feature names internally; mismatches raise
        #      warnings or wrong predictions.
        # ------------------------------------------------------------------
        expected_features = resources.feature_list
        expected_count = len(expected_features)

        # Check for missing features
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            raise FeatureMismatchError(
                f"Features present in training but missing from inference DataFrame: "
                f"{missing_features}"
            )

        # Select and reorder — any extra columns are silently dropped here
        # (e.g. we never added isFraud, so there are none, but be explicit)
        df = df[expected_features]

        # Final count validation
        if df.shape[1] != expected_count:
            raise FeatureMismatchError(
                f"Feature count mismatch after reorder: "
                f"expected {expected_count}, got {df.shape[1]}. "
                f"Columns: {df.columns.tolist()}"
            )

        logger.debug(
            "Feature vector prepared | txn=%s | features=%d",
            transaction_id,
            df.shape[1],
        )

        # ------------------------------------------------------------------
        # 7. Predict
        #
        #    predict_proba returns shape (n_samples, n_classes).
        #    Class 0 = legitimate, class 1 = fraud.
        #    We extract [:, 1] and take the first (only) row.
        # ------------------------------------------------------------------
        proba_array = resources.model.predict_proba(df)
        fraud_probability = float(proba_array[0, 1])  # native Python float

        # ------------------------------------------------------------------
        # 8. Generate prediction label
        # ------------------------------------------------------------------
        prediction = Prediction.FRAUD if fraud_probability >= 0.5 else Prediction.LEGITIMATE

        # ------------------------------------------------------------------
        # 9. Apply decision thresholds
        #
        #    APPROVE  fraud_probability < approve_threshold
        #    REVIEW   approve_threshold ≤ prob ≤ review_threshold
        #    BLOCK    fraud_probability > review_threshold
        # ------------------------------------------------------------------
        if fraud_probability < settings.approve_threshold:
            decision = Decision.APPROVE
        elif fraud_probability <= settings.review_threshold:
            decision = Decision.REVIEW
        else:
            decision = Decision.BLOCK

        # ------------------------------------------------------------------
        # 10. Measure latency
        # ------------------------------------------------------------------
        latency_ms = (time.perf_counter() - start) * 1000.0

        logger.info(
            "Prediction complete | txn=%s | prob=%.4f | label=%s | decision=%s | latency=%.2fms",
            transaction_id,
            fraud_probability,
            prediction.value,
            decision.value,
            latency_ms,
        )

        return PredictionResponse(
            transaction_id=transaction_id,
            fraud_probability=round(fraud_probability, 6),
            prediction=prediction,
            decision=decision,
            latency_ms=round(latency_ms, 3),
            model_version=resources.model_version,
        )


# Module-level singleton — stateless, safe to reuse across requests
predictor = Predictor()
