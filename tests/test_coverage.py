"""
Coverage unit tests for S4/S5:
  - test_interval_coverage_nominal: [q10, q90] coverage >= 0.75 (nominal 0.80, 5pp slack)
  - test_cqr_coverage: conformalized Y_lo yields P(Y >= Y_lo) >= 1 - alpha_cov - 0.05
"""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from src.data.synthetic import generate_workload
from src.data.preprocess import build_dense_feature_table, create_purged_time_splits
from src.models.quantile_xgb import (
    train_quantile_models,
    compute_interval_metrics,
    QuantileModelSuite,
)


def _make_dataset():
    """Helper: generate small synthetic dataset and split for testing."""
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)

    # Smaller workload for fast testing
    config["data"]["synthetic"]["n_extents"] = 100
    config["data"]["synthetic"]["n_windows"] = 500

    df_raw, meta = generate_workload(config, seed=42, workload_type="bursty")
    h = config["data"]["horizon_h"]
    feature_df = build_dense_feature_table(
        df_raw, meta["n_windows"], meta["n_extents"], h=h
    )
    splits = create_purged_time_splits(feature_df, meta["n_windows"], config)
    return splits, config


class TestIntervalCoverage:
    """Test that [q10, q90] prediction intervals achieve reasonable coverage."""

    def test_interval_coverage_nominal(self):
        """
        Train q10, q50, q90 on a small synthetic set and verify that
        the [q10, q90] interval coverage on Test >= 0.75 (nominal 0.80).
        """
        splits, config = _make_dataset()

        # Train only the levels needed for this test
        X_train = splits.train_df[splits.feature_cols].values
        y_train = splits.train_df[splits.target_col].values

        suite = train_quantile_models(
            X_train, y_train,
            quantile_levels=[0.10, 0.50, 0.90],
            xgb_params={"n_estimators": 100, "max_depth": 4},
            verbose=False,
        )

        X_test = splits.test_df[splits.feature_cols].values
        y_test = splits.test_df[splits.target_col].values

        metrics = compute_interval_metrics(
            suite, X_test, y_test, q_lo=0.10, q_hi=0.90
        )

        print(f"\n  Interval coverage: {metrics['coverage']:.3f} "
              f"(nominal {metrics['nominal_coverage']:.2f})")
        print(f"  Mean width: {metrics['mean_width']:.2f}")

        assert metrics["coverage"] >= 0.75, (
            f"Interval [q10, q90] coverage {metrics['coverage']:.3f} < 0.75"
        )


class TestCQRCoverage:
    """Test conformalized quantile regression coverage."""

    def test_cqr_coverage(self):
        """
        CQR score: E_i = q_lo(X_i) - Y_i on Cal-A.
        Q = ceil((n+1)(1-alpha_cov)) / n -th quantile of {E_i}.
        Y_lo(x) = q_lo(x) - Q, clamped >= 0.
        Assert: P(Y >= Y_lo) >= 1 - alpha_cov - 0.05 on Test.
        """
        splits, config = _make_dataset()
        alpha_cov = config["calibration"]["alpha_cov"]

        X_train = splits.train_df[splits.feature_cols].values
        y_train = splits.train_df[splits.target_col].values

        suite = train_quantile_models(
            X_train, y_train,
            quantile_levels=[0.10, 0.50, 0.90],
            xgb_params={"n_estimators": 100, "max_depth": 4},
            verbose=False,
        )

        # CQR calibration on Cal-A
        X_cal_a = splits.cal_a_df[splits.feature_cols].values
        y_cal_a = splits.cal_a_df[splits.target_col].values

        preds_cal_a = suite.predict_sorted(X_cal_a)
        q_lo_cal = preds_cal_a[0.10]

        # CQR scores: E_i = q_lo(X_i) - Y_i
        scores = q_lo_cal - y_cal_a
        n = len(scores)
        # Quantile index: ceil((n+1)(1-alpha)) / n
        quantile_level = np.ceil((n + 1) * (1 - alpha_cov)) / n
        quantile_level = min(quantile_level, 1.0)
        Q = float(np.quantile(scores, quantile_level))

        # Apply on Test
        X_test = splits.test_df[splits.feature_cols].values
        y_test = splits.test_df[splits.target_col].values

        preds_test = suite.predict_sorted(X_test)
        q_lo_test = preds_test[0.10]
        Y_lo = np.maximum(q_lo_test - Q, 0.0)

        # Check coverage: P(Y >= Y_lo) >= 1 - alpha_cov - 0.05
        coverage = float(np.mean(y_test >= Y_lo))
        target = 1 - alpha_cov - 0.05

        print(f"\n  CQR coverage P(Y >= Y_lo): {coverage:.3f}")
        print(f"  Target: >= {target:.3f} (1 - alpha_cov - 0.05)")
        print(f"  CQR offset Q: {Q:.4f}")

        assert coverage >= target, (
            f"CQR coverage {coverage:.3f} < {target:.3f}"
        )
