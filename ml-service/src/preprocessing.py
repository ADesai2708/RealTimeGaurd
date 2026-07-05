"""
preprocessing.py

Reusable preprocessing pipeline for the
RealTimeGuard fraud detection project.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
from src.feature_engineering import add_engineered_features


class DataPreprocessor:
    """
    Handles preprocessing for the fraud detection dataset.
    """

    def __init__(self):
        self.label_encoder = LabelEncoder()

    def preprocess(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """
        Preprocess the transaction dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.

        training : bool
            True during model training.
            False during inference.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe.
        """

        df = df.copy()

        # --------------------------------
        # Drop unnecessary columns
        # --------------------------------
        columns_to_drop = [
            "nameOrig",
            "nameDest",
            "isFlaggedFraud"
        ]

        df = df.drop(columns=columns_to_drop)

        # --------------------------------
        # Remove duplicate rows
        # --------------------------------
        df = df.drop_duplicates()

        # --------------------------------
        # Validate missing values
        # --------------------------------
        if df.isnull().sum().sum() != 0:
            raise ValueError("Dataset contains missing values.")

        # --------------------------------
        # Feature Engineering
        # --------------------------------
        df = add_engineered_features(df)

        # --------------------------------
        # Encode transaction type
        # --------------------------------
        if training:
            df["type"] = self.label_encoder.fit_transform(df["type"])
        else:
            df["type"] = self.label_encoder.transform(df["type"])

        return df
    def save_encoder(self, path: str):
        """
        Save fitted label encoder.
        """
        joblib.dump(self.label_encoder, path)

    def load_encoder(self, path: str):
        """
        Load previously fitted label encoder.
        """
        self.label_encoder = joblib.load(path)