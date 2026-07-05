"""
feature_engineering.py

Contains reusable feature engineering functions for the
RealTimeGuard fraud detection pipeline.
"""

import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the transaction dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction dataframe.

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
    # -----------------------------
    threshold = df["amount"].quantile(0.95)

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