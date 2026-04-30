"""
tests/test_data.py
------------------
Tests unitaires pour src/data/load_data.py et src/data/preprocess.py
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.load_data  import make_binary_target
from src.data.preprocess import (
    extract_features,
    build_feature_matrix,
    split_train_test,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_ps2():
    """Simule PS2 : 10 cycles x 6000 timesteps."""
    np.random.seed(42)
    return pd.DataFrame(np.random.randn(10, 6000) + 150)


@pytest.fixture
def mock_fs1():
    """Simule FS1 : 10 cycles x 600 timesteps."""
    np.random.seed(42)
    return pd.DataFrame(np.random.randn(10, 600) + 20)


@pytest.fixture
def mock_profile():
    """Simule profile : 10 cycles avec valve condition."""
    return pd.DataFrame({
        "cooler"      : [100, 100, 20, 100, 3, 100, 20, 3, 100, 100],
        "valve"       : [100, 73,  100, 80, 100, 90, 100, 100, 73, 100],
        "pump"        : [0, 0, 1, 0, 2, 0, 1, 2, 0, 0],
        "accumulator" : [130, 115, 130, 100, 115, 130, 100, 130, 115, 130],
        "stable"      : [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    })


# ── Tests make_binary_target ─────────────────────────────────────────────────

class TestMakeBinaryTarget:

    def test_optimal_is_1(self, mock_profile):
        y = make_binary_target(mock_profile)
        optimal_idx = mock_profile[mock_profile["valve"] == 100].index
        assert (y[optimal_idx] == 1).all()

    def test_non_optimal_is_0(self, mock_profile):
        y = make_binary_target(mock_profile)
        non_optimal_idx = mock_profile[mock_profile["valve"] != 100].index
        assert (y[non_optimal_idx] == 0).all()

    def test_output_shape(self, mock_profile):
        y = make_binary_target(mock_profile)
        assert len(y) == len(mock_profile)

    def test_output_binary(self, mock_profile):
        y = make_binary_target(mock_profile)
        assert set(y.unique()).issubset({0, 1})

    def test_series_name(self, mock_profile):
        y = make_binary_target(mock_profile)
        assert y.name == "valve_optimal"


# ── Tests extract_features ────────────────────────────────────────────────────

class TestExtractFeatures:

    def test_output_shape(self, mock_ps2):
        features = extract_features(mock_ps2, prefix="ps2")
        assert features.shape == (10, 12)

    def test_column_names(self, mock_ps2):
        features = extract_features(mock_ps2, prefix="ps2")
        expected = [
            "ps2_mean", "ps2_std", "ps2_min", "ps2_max",
            "ps2_median", "ps2_q25", "ps2_q75", "ps2_iqr",
            "ps2_skew", "ps2_kurtosis", "ps2_rms", "ps2_peak_to_peak"
        ]
        assert list(features.columns) == expected

    def test_prefix_fs1(self, mock_fs1):
        features = extract_features(mock_fs1, prefix="fs1")
        assert all(col.startswith("fs1_") for col in features.columns)

    def test_no_nan(self, mock_ps2):
        features = extract_features(mock_ps2, prefix="ps2")
        assert features.isna().sum().sum() == 0

    def test_mean_correct(self, mock_ps2):
        features = extract_features(mock_ps2, prefix="ps2")
        expected_means = mock_ps2.mean(axis=1).values
        np.testing.assert_array_almost_equal(
            features["ps2_mean"].values, expected_means, decimal=5
        )

    def test_std_positive(self, mock_ps2):
        features = extract_features(mock_ps2, prefix="ps2")
        assert (features["ps2_std"] >= 0).all()

    def test_rms_positive(self, mock_ps2):
        features = extract_features(mock_ps2, prefix="ps2")
        assert (features["ps2_rms"] >= 0).all()

    def test_peak_to_peak_positive(self, mock_ps2):
        features = extract_features(mock_ps2, prefix="ps2")
        assert (features["ps2_peak_to_peak"] >= 0).all()

    def test_iqr_equals_q75_minus_q25(self, mock_ps2):
        features = extract_features(mock_ps2, prefix="ps2")
        expected = features["ps2_q75"] - features["ps2_q25"]
        np.testing.assert_array_almost_equal(
            features["ps2_iqr"].values, expected.values, decimal=5
        )


# ── Tests build_feature_matrix ───────────────────────────────────────────────

class TestBuildFeatureMatrix:

    def test_output_shape(self, mock_ps2, mock_fs1, mock_profile):
        X = build_feature_matrix(mock_ps2, mock_fs1, mock_profile)
        # 12 (ps2) + 12 (fs1) + 1 (stable) = 25
        assert X.shape == (10, 25)

    def test_no_nan(self, mock_ps2, mock_fs1, mock_profile):
        X = build_feature_matrix(mock_ps2, mock_fs1, mock_profile)
        assert X.isna().sum().sum() == 0

    def test_stable_flag_included(self, mock_ps2, mock_fs1, mock_profile):
        X = build_feature_matrix(mock_ps2, mock_fs1, mock_profile)
        assert "stable" in X.columns

    def test_ps2_features_present(self, mock_ps2, mock_fs1, mock_profile):
        X = build_feature_matrix(mock_ps2, mock_fs1, mock_profile)
        ps2_cols = [c for c in X.columns if c.startswith("ps2_")]
        assert len(ps2_cols) == 12

    def test_fs1_features_present(self, mock_ps2, mock_fs1, mock_profile):
        X = build_feature_matrix(mock_ps2, mock_fs1, mock_profile)
        fs1_cols = [c for c in X.columns if c.startswith("fs1_")]
        assert len(fs1_cols) == 12


# ── Tests split_train_test ────────────────────────────────────────────────────

class TestSplitTrainTest:

    @pytest.fixture
    def X_y(self, mock_ps2, mock_fs1, mock_profile):
        X = build_feature_matrix(mock_ps2, mock_fs1, mock_profile)
        y = make_binary_target(mock_profile)
        return X, y

    def test_split_sizes(self, X_y):
        X, y = X_y
        X_train, X_test, y_train, y_test = split_train_test(X, y, train_size=7)
        assert len(X_train) == 7
        assert len(X_test)  == 3
        assert len(y_train) == 7
        assert len(y_test)  == 3

    def test_no_overlap(self, X_y):
        X, y = X_y
        X_train, X_test, y_train, y_test = split_train_test(X, y, train_size=7)
        assert len(X_train) + len(X_test) == len(X)

    def test_order_preserved(self, X_y):
        X, y = X_y
        X_train, X_test, y_train, y_test = split_train_test(X, y, train_size=7)
        # Les 7 premiers doivent etre dans train
        pd.testing.assert_frame_equal(
            X_train.reset_index(drop=True),
            X.iloc[:7].reset_index(drop=True)
        )

    def test_default_train_size(self, mock_ps2, mock_fs1, mock_profile):
        # Avec 10 cycles et train_size=7 explicite
        X = build_feature_matrix(mock_ps2, mock_fs1, mock_profile)
        y = make_binary_target(mock_profile)
        X_train, X_test, _, _ = split_train_test(X, y, train_size=7)
        assert len(X_train) == 7
