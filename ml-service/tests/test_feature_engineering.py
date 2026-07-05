"""
tests/test_feature_engineering.py

Unit tests for src/feature_engineering.py

Uses a minimal synthetic DataFrame — no dependency on PaySim dataset.
"""

import pandas as pd
import pytest

from src.feature_engineering import add_engineered_features


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal raw transaction DataFrame."""
    return pd.DataFrame(
        {
            "step":           [1, 25, 48, 100],
            "amount":         [200.0, 5000.0, 100.0, 9000.0],
            "oldbalanceOrg":  [1000.0, 5000.0, 0.0, 200.0],
            "newbalanceOrig": [800.0, 0.0, 0.0, 0.0],
            "oldbalanceDest": [0.0, 200.0, 1000.0, 0.0],
            "newbalanceDest": [200.0, 5200.0, 900.0, 9000.0],
        }
    )


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

class TestAddEngineeredFeatures:

    def test_returns_dataframe(self, sample_df):
        result = add_engineered_features(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_input(self, sample_df):
        original_cols = set(sample_df.columns)
        _ = add_engineered_features(sample_df)
        assert set(sample_df.columns) == original_cols

    def test_all_seven_features_added(self, sample_df):
        result = add_engineered_features(sample_df)
        expected_new_cols = {
            "balanceChange",
            "destinationBalanceChange",
            "hour",
            "day",
            "isLargeTransaction",
            "originBalanceZero",
            "destinationBalanceZero",
        }
        assert expected_new_cols.issubset(set(result.columns))

    def test_balance_change_calculation(self, sample_df):
        result = add_engineered_features(sample_df)
        # balanceChange = oldbalanceOrg - newbalanceOrig
        expected = sample_df["oldbalanceOrg"] - sample_df["newbalanceOrig"]
        pd.testing.assert_series_equal(
            result["balanceChange"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_destination_balance_change_calculation(self, sample_df):
        result = add_engineered_features(sample_df)
        expected = sample_df["newbalanceDest"] - sample_df["oldbalanceDest"]
        pd.testing.assert_series_equal(
            result["destinationBalanceChange"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_hour_is_step_mod_24(self, sample_df):
        result = add_engineered_features(sample_df)
        expected = sample_df["step"] % 24
        pd.testing.assert_series_equal(
            result["hour"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_day_is_step_div_24(self, sample_df):
        result = add_engineered_features(sample_df)
        expected = sample_df["step"] // 24
        pd.testing.assert_series_equal(
            result["day"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_is_large_transaction_is_binary(self, sample_df):
        result = add_engineered_features(sample_df)
        assert set(result["isLargeTransaction"].unique()).issubset({0, 1})

    def test_origin_balance_zero_flag(self, sample_df):
        result = add_engineered_features(sample_df)
        # Row 2 has oldbalanceOrg == 0
        assert result.loc[2, "originBalanceZero"] == 1
        assert result.loc[0, "originBalanceZero"] == 0

    def test_destination_balance_zero_flag(self, sample_df):
        result = add_engineered_features(sample_df)
        # Row 0 and row 3 have oldbalanceDest == 0
        assert result.loc[0, "destinationBalanceZero"] == 1
        assert result.loc[3, "destinationBalanceZero"] == 1
        assert result.loc[1, "destinationBalanceZero"] == 0

    def test_no_null_values_introduced(self, sample_df):
        result = add_engineered_features(sample_df)
        assert result.isnull().sum().sum() == 0
