"""
Preprocessing and feature engineering module for ISM.

Per-extent per-window aggregation:
- Target Y: sum of accesses in the next h windows (decision horizon h=5).
- Candidate set: only extents observed in past windows.
- Features:
  - Multi-lags: lag_1 ... lag_8
  - EWMAs: span 5, span 10
  - Burstiness: CV (std / mean) over past k=8 windows
  - Read/write ratio and mean request size
  - Neighbour extent activity (extent - 1, extent + 1)
  - Volume-level total IOPS in current window
- 4-way time-ordered split with purge gaps of h windows:
  - Train: 40%
  - Cal-A: 10%
  - Cal-B: 30%
  - Test: 20%
- Decision epochs: non-overlapping every h windows.
- RobustScaler fit strictly on Train.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


FEATURE_COLUMNS = [
    "count_lag1", "count_lag2", "count_lag3", "count_lag4",
    "count_lag5", "count_lag6", "count_lag7", "count_lag8",
    "ewma_5", "ewma_10",
    "burstiness_8",
    "read_ratio",
    "mean_size_kb",
    "neighbour_prev_count",
    "neighbour_next_count",
    "volume_load",
]


@dataclass
class DatasetSplits:
    train_df: pd.DataFrame
    cal_a_df: pd.DataFrame
    cal_b_df: pd.DataFrame
    test_df: pd.DataFrame
    scaler: RobustScaler
    feature_cols: List[str]
    target_col: str
    metadata: Dict[str, Any]


def build_dense_feature_table(
    df: pd.DataFrame,
    n_windows: int,
    n_extents: int,
    h: int = 5,
    max_lag: int = 8,
    extent_size_bytes: int = 1024 * 1024,
) -> pd.DataFrame:
    """
    Build feature matrix across all (window, extent) pairs where the extent
    has been observed in past windows (preventing future extent leakage).
    """
    # 2D dense matrices for fast vectorized feature calculation
    counts = np.zeros((n_windows, n_extents), dtype=np.float32)
    read_counts = np.zeros((n_windows, n_extents), dtype=np.float32)
    mean_sizes = np.zeros((n_windows, n_extents), dtype=np.float32)

    w_idx = df["window"].values
    e_idx = df["extent"].values
    counts[w_idx, e_idx] = df["count"].values
    read_counts[w_idx, e_idx] = df["read_count"].values
    mean_sizes[w_idx, e_idx] = df["mean_size_bytes"].values / 1024.0  # KiB

    # Volume-level total load per window
    vol_load = counts.sum(axis=1)  # (n_windows,)

    # Multi-lag matrices (n_windows, n_extents)
    lags = {}
    for lag in range(1, max_lag + 1):
        lag_mat = np.zeros_like(counts)
        lag_mat[lag:, :] = counts[:-lag, :]
        lags[f"count_lag{lag}"] = lag_mat

    # EWMAs along window axis (default axis=0)
    df_counts = pd.DataFrame(counts)
    ewma_5 = df_counts.ewm(span=5).mean().values.astype(np.float32)
    ewma_10 = df_counts.ewm(span=10).mean().values.astype(np.float32)

    # Burstiness: CV of per-window counts over last k=8 windows
    roll_mean_8 = df_counts.rolling(window=8, min_periods=1).mean().values
    roll_std_8 = df_counts.rolling(window=8, min_periods=1).std(ddof=0).values
    burstiness_8 = (roll_std_8 / (roll_mean_8 + 1e-5)).astype(np.float32)

    # Neighbour extent activity in current window: extent - 1, extent + 1
    neigh_prev = np.zeros_like(counts)
    neigh_next = np.zeros_like(counts)
    neigh_prev[:, 1:] = counts[:, :-1]
    neigh_next[:, :-1] = counts[:, 1:]

    # Read ratio
    read_ratio = (read_counts / (counts + 1e-5)).astype(np.float32)

    # Target Y: sum of counts in windows [w + 1, w + h]
    # For window w, next h windows are w+1 .. w+h
    Y_target = np.zeros((n_windows, n_extents), dtype=np.float32)
    for k in range(1, h + 1):
        Y_target[: n_windows - k, :] += counts[k:, :]

    # Track first observed window per extent for candidate filtering
    has_activity = (counts > 0)
    first_observed = np.full(n_extents, n_windows, dtype=np.int32)
    for e in range(n_extents):
        act = np.where(has_activity[:, e])[0]
        if len(act) > 0:
            first_observed[e] = act[0]

    # Collect decision epochs: non-overlapping every h windows
    # Start after max_lag so lags are populated, and stop h windows before end
    epoch_windows = np.arange(max_lag, n_windows - h, h, dtype=np.int32)

    # Vectorized candidate mask: extents observed at or before window w.
    window_grid = epoch_windows[:, None]
    extent_grid = np.arange(n_extents, dtype=np.int32)[None, :]
    active = first_observed[None, :] <= window_grid
    wi, ei = np.where(active)
    w_idx = epoch_windows[wi]
    e_idx = ei.astype(np.int32)

    data = {
        "window": w_idx,
        "extent": e_idx,
        "volume_load": vol_load[w_idx],
        "read_ratio": read_ratio[w_idx, e_idx],
        "mean_size_kb": mean_sizes[w_idx, e_idx],
        "ewma_5": ewma_5[w_idx, e_idx],
        "ewma_10": ewma_10[w_idx, e_idx],
        "burstiness_8": burstiness_8[w_idx, e_idx],
        "neighbour_prev_count": neigh_prev[w_idx, e_idx],
        "neighbour_next_count": neigh_next[w_idx, e_idx],
        "Y": Y_target[w_idx, e_idx],
    }
    for lag in range(1, max_lag + 1):
        data[f"count_lag{lag}"] = lags[f"count_lag{lag}"][w_idx, e_idx]

    feature_table = pd.DataFrame(data)
    feature_table.attrs["extent_size_bytes"] = int(extent_size_bytes)
    return feature_table


def create_purged_time_splits(
    feature_df: pd.DataFrame,
    n_windows: int,
    config: Dict[str, Any],
) -> DatasetSplits:
    """
    Split feature table into 4 partitions with purge gaps of h windows:
    - Train: 40%
    - Purge gap: h windows
    - Cal-A: 10%
    - Purge gap: h windows
    - Cal-B: 30%
    - Purge gap: h windows
    - Test: 20%
    """
    h = config["data"].get("horizon_h", 5)
    splits_cfg = config.get("splits", {})
    train_frac = splits_cfg.get("train", 0.40)
    cal_a_frac = splits_cfg.get("cal_a", 0.10)
    cal_b_frac = splits_cfg.get("cal_b", 0.30)
    test_frac = splits_cfg.get("test", 0.20)

    # Window boundaries
    train_end = int(n_windows * train_frac)
    cal_a_start = train_end + h
    cal_a_end = cal_a_start + int(n_windows * cal_a_frac)
    cal_b_start = cal_a_end + h
    cal_b_end = cal_b_start + int(n_windows * cal_b_frac)
    test_start = cal_b_end + h
    test_end = n_windows

    # Partition filtering by window index
    train_df = feature_df[feature_df["window"] < train_end].copy().reset_index(drop=True)
    cal_a_df = feature_df[(feature_df["window"] >= cal_a_start) & (feature_df["window"] < cal_a_end)].copy().reset_index(drop=True)
    cal_b_df = feature_df[(feature_df["window"] >= cal_b_start) & (feature_df["window"] < cal_b_end)].copy().reset_index(drop=True)
    test_df = feature_df[feature_df["window"] >= test_start].copy().reset_index(drop=True)

    # Fit RobustScaler strictly on Train
    scaler = RobustScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])

    # Transform feature columns (prefixed with 'scaled_')
    scaled_feature_cols = [f"s_{col}" for col in FEATURE_COLUMNS]
    for split_df in [train_df, cal_a_df, cal_b_df, test_df]:
        scaled_vals = scaler.transform(split_df[FEATURE_COLUMNS])
        for idx, scol in enumerate(scaled_feature_cols):
            split_df[scol] = scaled_vals[:, idx]

    metadata = {
        "n_windows": n_windows,
        "horizon_h": h,
        "boundaries": {
            "train": [0, train_end],
            "cal_a": [cal_a_start, cal_a_end],
            "cal_b": [cal_b_start, cal_b_end],
            "test": [test_start, test_end],
        },
        "rows": {
            "train": len(train_df),
            "cal_a": len(cal_a_df),
            "cal_b": len(cal_b_df),
            "test": len(test_df),
        },
        "purge_gap_windows": h,
    }

    return DatasetSplits(
        train_df=train_df,
        cal_a_df=cal_a_df,
        cal_b_df=cal_b_df,
        test_df=test_df,
        scaler=scaler,
        feature_cols=scaled_feature_cols,
        target_col="Y",
        metadata=metadata,
    )
