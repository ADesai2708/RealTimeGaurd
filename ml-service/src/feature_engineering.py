"""
feature_engineering.py

Contains reusable feature engineering functions for the
RealTimeGuard fraud detection pipeline.
"""

from __future__ import annotations

import pandas as pd


def add_engineered_features(
    df: pd.DataFrame,
    large_transaction_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Add engineered features to the transaction dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction dataframe.
    large_transaction_threshold : float or None
        The pre-fitted 95th-percentile amount threshold used to flag
        large transactions.

        - **Training path** (``None``): the threshold is computed from
          ``df["amount"].quantile(0.95)``.  Pass ``None`` during training
          so the threshold is derived from the full training distribution.
        - **Inference path** (float): pass the threshold that was computed
          and persisted during training.  This prevents training-serving
          skew: a single-row inference DataFrame would otherwise compute
          ``quantile(0.95)`` equal to its own ``amount``, making
          ``isLargeTransaction`` always ``0``.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered features.
    """

    # Create a copy to avoid modifying the original dataframe
    df = df.copy()

    # -----------------------------
    # Balance change (origin account)
    # -----------------------------
    df["balanceChange"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"]
    )

    # -----------------------------
    # Balance change (destination account)
    # -----------------------------
    df["destinationBalanceChange"] = (
        df["newbalanceDest"] - df["oldbalanceDest"]
    )

    # -----------------------------
    # Hour of day
    # -----------------------------
    df["hour"] = df["step"] % 24

    # -----------------------------
    # Day number
    # -----------------------------
    df["day"] = df["step"] // 24

    # -----------------------------
    # Large transaction flag
    #
    # IMPORTANT: During training, `large_transaction_threshold` is None and
    # the threshold is computed from the full training distribution (correct).
    # During inference, the caller MUST supply the persisted training-time
    # threshold to avoid computing quantile on a 1-row DataFrame (which would
    # always equal the row's own amount, making this flag perpetually 0).
    # -----------------------------
    if large_transaction_threshold is None:
        # Training path: compute from the dataset distribution
        threshold = df["amount"].quantile(0.95)
    else:
        # Inference path: use the pre-fitted value from preprocessing_config.json
        threshold = large_transaction_threshold

    df["isLargeTransaction"] = (
        df["amount"] > threshold
    ).astype(int)

    # -----------------------------
    # Origin balance zero
    # -----------------------------
    df["originBalanceZero"] = (
        df["oldbalanceOrg"] == 0
    ).astype(int)

    # -----------------------------
    # Destination balance zero
    # -----------------------------
    df["destinationBalanceZero"] = (
        df["oldbalanceDest"] == 0
    ).astype(int)

    return df