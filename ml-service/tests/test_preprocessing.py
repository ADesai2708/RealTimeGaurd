"""
tests/test_preprocessing.py

Unit tests for src/preprocessing.py

Uses a minimal synthetic DataFrame — no dependency on PaySim dataset.
"""

import pandas as pd
import pytest

from src.preprocessing import DataPreprocessor


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

def _make_raw_df(n: int = 8) -> pd.DataFrame:
    """
    Build a minimal raw DataFrame matching PaySim column structure,
    with balanced TRANSFER and CASH_OUT types.
    """
    types = ["TRANSFER", "CASH_OUT"] * (n // 2)
    return pd.DataFrame(
        {
            "step":            list(range(n)),
            "type":            types,
            "amount":          [float(i * 100 + 50) for i in range(n)],
            "nameOrig":        [f"C{i:05d}" for i in range(n)],
            "oldbalanceOrg":   [1000.0] * n,
            "newbalanceOrig":  [900.0] * n,
            "nameDest":        [f"M{i:05d}" for i in range(n)],
            "oldbalanceDest":  [0.0] * n,
            "newbalanceDest":  [100.0] * n,
            "isFraud":         [0, 1] * (n // 2),
            "isFlaggedFraud":  [0] * n,
        }
    )


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return _make_raw_df(n=8)


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    return DataPreprocessor()


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

class TestDataPreprocessor:

    def test_returns_dataframe(self, preprocessor, raw_df):
        result = preprocessor.preprocess(raw_df, training=True)
        assert isinstance(result, pd.DataFrame)

    def test_drops_name_columns(self, preprocessor, raw_df):
        result = preprocessor.preprocess(raw_df, training=True)
        assert "nameOrig" not in result.columns
        assert "nameDest" not in result.columns

    def test_drops_is_flagged_fraud(self, preprocessor, raw_df):
        result = preprocessor.preprocess(raw_df, training=True)
        assert "isFlaggedFraud" not in result.columns

    def test_removes_duplicates(self, preprocessor, raw_df):
        df_with_dupes = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
        result = preprocessor.preprocess(df_with_dupes, training=True)
        # Duplicate row was removed
        assert len(result) == len(raw_df)

    def test_raises_on_missing_values(self, preprocessor, raw_df):
        raw_df.loc[0, "amount"] = float("nan")
        with pytest.raises(ValueError, match="missing values"):
            preprocessor.preprocess(raw_df, training=True)

    def test_type_column_is_encoded(self, preprocessor, raw_df):
        result = preprocessor.preprocess(raw_df, training=True)
        assert result["type"].dtype in [int, "int64", "int32"]

    def test_engineered_features_present(self, preprocessor, raw_df):
        result = preprocessor.preprocess(raw_df, training=True)
        for col in ["balanceChange", "destinationBalanceChange", "hour", "day",
                    "isLargeTransaction", "originBalanceZero", "destinationBalanceZero"]:
            assert col in result.columns, f"Missing engineered feature: {col}"

    def test_does_not_mutate_input(self, preprocessor, raw_df):
        original_cols = set(raw_df.columns)
        _ = preprocessor.preprocess(raw_df, training=True)
        assert set(raw_df.columns) == original_cols

    def test_inference_mode_uses_fitted_encoder(self, preprocessor, raw_df):
        # Fit on training data
        preprocessor.preprocess(raw_df, training=True)
        # Now use the same types for inference
        result = preprocessor.preprocess(raw_df, training=False)
        assert result["type"].dtype in [int, "int64", "int32"]

    def test_encoder_save_and_load(self, preprocessor, raw_df, tmp_path):
        preprocessor.preprocess(raw_df, training=True)
        encoder_path = tmp_path / "encoder.pkl"

        preprocessor.save_encoder(encoder_path)
        assert encoder_path.exists()

        # Load into a new preprocessor and use it
        new_preprocessor = DataPreprocessor()
        new_preprocessor.load_encoder(encoder_path)
        result = new_preprocessor.preprocess(raw_df, training=False)
        assert result["type"].dtype in [int, "int64", "int32"]
