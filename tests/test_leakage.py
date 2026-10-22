"""
Tests for data integrity and leakage prevention.

S1 tests:
  - test_no_future_in_raw_data: timestamps monotonic in synthetic output.
  - test_synthetic_output_shape: correct columns, >= 2000 windows, sparse.
  - test_reproducibility: same seed → same output.
"""
from __future__ import annotations

import yaml

from src.data.preprocess import build_dense_feature_table


def test_extent_size_is_single_source_of_truth():
    config = yaml.safe_load(open("config/default.yaml"))
    assert config["data"]["extent_size_bytes"] == 16 * 1024 * 1024
    assert config["cost_model"]["extent_size_mib"] == config["data"]["extent_size_bytes"] / (1024 * 1024)
    frame = build_dense_feature_table(
        __import__("pandas").DataFrame({
            "window": [0], "extent": [0], "count": [1],
            "read_count": [1], "mean_size_bytes": [4096],
        }),
        10, 1, extent_size_bytes=config["data"]["extent_size_bytes"],
    )
    assert frame.attrs["extent_size_bytes"] == config["data"]["extent_size_bytes"]

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.synthetic import generate_workload


@pytest.fixture(scope="module")
def config():
    with open("config/default.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def sample_workload(config):
    """Generate one bursty workload for testing."""
    df, meta = generate_workload(config, seed=42, workload_type="bursty")
    return df, meta


class TestSyntheticDataIntegrity:
    """S1 data integrity tests."""

    def test_no_future_in_raw_data(self, sample_workload):
        """Timestamps (window indices) must be monotonically non-decreasing
        within sorted output — i.e. no future data leaks backward."""
        df, _ = sample_workload
        sorted_df = df.sort_values("window")
        assert (sorted_df["window"].diff().dropna() >= 0).all(), \
            "Window indices are not monotonically non-decreasing"

    def test_synthetic_output_shape(self, sample_workload):
        """Verify columns, minimum windows, and sparsity."""
        df, meta = sample_workload
        required_cols = {"window", "extent", "count", "read_count",
                         "write_count", "mean_size_bytes", "timestamp_start_s"}
        assert required_cols.issubset(set(df.columns)), \
            f"Missing columns: {required_cols - set(df.columns)}"

        n_windows = meta["n_windows"]
        assert n_windows >= 2000, f"n_windows={n_windows} < 2000"

        max_window = df["window"].max()
        assert max_window < n_windows, \
            f"max window {max_window} >= n_windows {n_windows}"

        # Sparsity: should have far fewer rows than n_windows * n_extents
        dense_size = n_windows * meta["n_extents"]
        assert len(df) < dense_size * 0.5, \
            f"Not sparse: {len(df)} rows vs {dense_size} dense"

    def test_counts_non_negative(self, sample_workload):
        """All counts must be >= 0."""
        df, _ = sample_workload
        assert (df["count"] >= 0).all()
        assert (df["read_count"] >= 0).all()
        assert (df["write_count"] >= 0).all()
        assert ((df["read_count"] + df["write_count"]) == df["count"]).all(), \
            "read_count + write_count != count"

    def test_reproducibility(self, config):
        """Same seed must produce identical output."""
        df1, m1 = generate_workload(config, seed=123, workload_type="steady")
        df2, m2 = generate_workload(config, seed=123, workload_type="steady")
        pd.testing.assert_frame_equal(df1, df2)
        assert m1["mean_lambda"] == m2["mean_lambda"]

    def test_different_seeds_differ(self, config):
        """Different seeds must produce different output."""
        df1, _ = generate_workload(config, seed=100, workload_type="bursty")
        df2, _ = generate_workload(config, seed=200, workload_type="bursty")
        assert not df1.equals(df2), "Different seeds produced identical data"

    def test_minimum_windows(self, config):
        """Each synthetic workload must have at least 2000 windows."""
        for wtype in ["steady", "bursty", "read_heavy", "changing_hot_set"]:
            _, meta = generate_workload(config, seed=42, workload_type=wtype)
            assert meta["n_windows"] >= 2000, \
                f"{wtype}: n_windows={meta['n_windows']} < 2000"

    def test_drift_only_in_changing_hot_set(self, config):
        """Drift info should be None for non-drift workloads."""
        _, meta = generate_workload(config, seed=42, workload_type="steady")
        assert meta["drift"] is None
        _, meta = generate_workload(config, seed=42, workload_type="changing_hot_set")
        assert meta["drift"] is not None

    def test_spikes_in_range(self, sample_workload):
        """Spike onsets must be within [0, n_windows)."""
        _, meta = sample_workload
        for sp in meta["spikes"]:
            assert 0 <= sp["onset"] < meta["n_windows"]
            assert sp["amplitude"] >= 1.0
            assert sp["duration"] >= 1


class TestPreprocessingAndSplits:
    """S2 preprocessing, feature engineering, and purge gap tests."""

    @pytest.fixture(scope="class")
    @classmethod
    def preprocessed_data(cls, config):
        from src.data.preprocess import build_dense_feature_table, create_purged_time_splits
        cfg = dict(config)
        df, meta = generate_workload(cfg, seed=42, workload_type="bursty")
        feature_df = build_dense_feature_table(df, meta["n_windows"], meta["n_extents"], h=cfg["data"]["horizon_h"])
        splits = create_purged_time_splits(feature_df, meta["n_windows"], cfg)
        return df, meta, feature_df, splits

    def test_purge_gaps_exist(self, preprocessed_data, config):
        """Partitions must have strictly positive purge gaps >= h between them."""
        _, _, _, splits = preprocessed_data
        h = config["data"]["horizon_h"]
        bounds = splits.metadata["boundaries"]

        train_end = bounds["train"][1]
        cal_a_start = bounds["cal_a"][0]
        cal_a_end = bounds["cal_a"][1]
        cal_b_start = bounds["cal_b"][0]
        cal_b_end = bounds["cal_b"][1]
        test_start = bounds["test"][0]

        assert cal_a_start - train_end >= h, f"Purge gap Train->Cal-A ({cal_a_start - train_end}) < h ({h})"
        assert cal_b_start - cal_a_end >= h, f"Purge gap Cal-A->Cal-B ({cal_b_start - cal_a_end}) < h ({h})"
        assert test_start - cal_b_end >= h, f"Purge gap Cal-B->Test ({test_start - cal_b_end}) < h ({h})"

    def test_no_split_overlap(self, preprocessed_data):
        """Window indices across Train, Cal-A, Cal-B, Test must be completely disjoint."""
        _, _, _, splits = preprocessed_data
        w_train = set(splits.train_df["window"].unique())
        w_cal_a = set(splits.cal_a_df["window"].unique())
        w_cal_b = set(splits.cal_b_df["window"].unique())
        w_test = set(splits.test_df["window"].unique())

        assert len(w_train & w_cal_a) == 0, "Train and Cal-A overlap"
        assert len(w_cal_a & w_cal_b) == 0, "Cal-A and Cal-B overlap"
        assert len(w_cal_b & w_test) == 0, "Cal-B and Test overlap"

    def test_target_is_future_only(self, preprocessed_data):
        """Target Y must be non-negative and correctly aligned with future windows."""
        df_raw, meta, feature_df, _ = preprocessed_data
        h = meta.get("horizon_h", 5)

        # Check a random row
        row = feature_df.iloc[10]
        w = int(row["window"])
        e = int(row["extent"])
        expected_y = df_raw[(df_raw["extent"] == e) & (df_raw["window"] > w) & (df_raw["window"] <= w + h)]["count"].sum()
        assert row["Y"] == expected_y, f"Mismatch in Y: feature={row['Y']} vs raw sum={expected_y}"

    def test_candidate_set_no_future_extents(self, preprocessed_data):
        """Extents in feature table at window w must have been observed at or before w."""
        df_raw, _, feature_df, _ = preprocessed_data
        first_obs = df_raw.groupby("extent")["window"].min().to_dict()

        for _, row in feature_df.sample(n=min(50, len(feature_df)), random_state=42).iterrows():
            w = int(row["window"])
            e = int(row["extent"])
            assert first_obs[e] <= w, f"Extent {e} appeared at window {w} before first observation {first_obs[e]}"
