"""
Dataset configuration loaders for ISM.

Supports three benchmark dataset modes:
1. 'synthetic': Seeded synthetic storage workload (Zipf + diurnal + spikes + decoys)
2. 'msrc': Pure Microsoft Research Cambridge enterprise storage trace (hm_0, prn_0, proj_0)
3. 'hybrid': Realistic enterprise cross-domain scenario:
   Trains on Synthetic + MSRC representations, calibrates on mixed domain,
   and tests against enterprise production workloads.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.data.preprocess import (
    FEATURE_COLUMNS,
    DatasetSplits,
    build_dense_feature_table,
    create_purged_time_splits,
)
from src.data.synthetic import generate_workload


def window_aggregate_trace(df: pd.DataFrame, window_sec: float) -> pd.DataFrame:
    """Aggregate request-level trace into per-(window, extent) stats."""
    t0 = float(df["timestamp_s"].min())
    w = ((df["timestamp_s"] - t0) / window_sec).astype(np.int32)
    out = df.copy()
    out["window"] = w
    grp = out.groupby(["window", "extent"], sort=False)
    agg = grp.agg(
        count=("extent", "size"),
        read_count=("is_read", "sum"),
        mean_size_bytes=("size_bytes", "mean"),
    ).reset_index()
    agg["write_count"] = agg["count"] - agg["read_count"]
    agg["timestamp_start_s"] = t0 + agg["window"] * window_sec
    return agg


def remap_trace_extents(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Map sparse physical extent IDs into contiguous 0..N-1 indices."""
    uniq = np.sort(df["extent"].unique())
    mapping = {int(e): i for i, e in enumerate(uniq)}
    out = df.copy()
    out["extent"] = out["extent"].map(mapping).astype(np.int32)
    return out, len(uniq)


def load_msrc_feature_splits(
    config: Dict[str, Any],
    volume: str = "hm_0",
    max_windows: int = 1500,
) -> Tuple[DatasetSplits, Dict[str, Any]]:
    """Load and process pure MSRC storage trace."""
    cache_dir = Path(config["data"]["msrc"]["cache_dir"])
    parquet_path = cache_dir / f"{volume}.parquet"
    if not parquet_path.exists():
        # Fallback search
        root = Path(__file__).resolve().parents[2]
        parquet_path = root / "data" / "msrc_parquet" / f"{volume}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"MSRC parquet trace file not found at {parquet_path}")

    raw = pd.read_parquet(parquet_path)
    window_sec = float(config["data"]["window_sec"])
    agg = window_aggregate_trace(raw, window_sec)
    agg, n_ext = remap_trace_extents(agg)

    n_windows = int(agg["window"].max()) + 1
    if n_windows > max_windows:
        agg = agg[agg["window"] < max_windows].copy()
        n_windows = max_windows

    features = build_dense_feature_table(
        agg,
        n_windows,
        n_ext,
        h=config["data"]["horizon_h"],
        extent_size_bytes=int(config["data"]["extent_size_bytes"]),
    )
    splits = create_purged_time_splits(features, n_windows, config)
    meta = {
        "dataset_mode": "msrc",
        "volume": volume,
        "n_windows": n_windows,
        "n_extents": n_ext,
        "extent_size_bytes": int(config["data"]["extent_size_bytes"]),
    }
    return splits, meta


def load_synthetic_feature_splits(
    config: Dict[str, Any],
    seed: int = 42,
    workload_type: str = "noisy_bursty",
) -> Tuple[DatasetSplits, Dict[str, Any]]:
    """Generate and process pure synthetic workload."""
    raw, meta = generate_workload(config, seed=seed, workload_type=workload_type)
    features = build_dense_feature_table(
        raw,
        meta["n_windows"],
        meta["n_extents"],
        h=config["data"]["horizon_h"],
        extent_size_bytes=int(config["data"]["extent_size_bytes"]),
    )
    splits = create_purged_time_splits(features, meta["n_windows"], config)
    meta["dataset_mode"] = "synthetic"
    return splits, meta


def load_hybrid_feature_splits(
    config: Dict[str, Any],
    seed: int = 42,
    volume: str = "hm_0",
) -> Tuple[DatasetSplits, Dict[str, Any]]:
    """
    Generate Hybrid dataset splits:
    Combines Synthetic data and Microsoft trace data into a unified,
    cross-domain training and evaluation environment.
    Train: balanced mix of synthetic and early MSRC patterns.
    Cal-A & Cal-B: pooled calibration containing both domains.
    Test: real MSRC test horizon + synthetic stress horizon.
    """
    syn_splits, syn_meta = load_synthetic_feature_splits(config, seed=seed, workload_type="bursty")
    msrc_splits, msrc_meta = load_msrc_feature_splits(config, volume=volume, max_windows=1500)

    # Combine partitions
    def concat_and_align(df_syn: pd.DataFrame, df_msrc: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in df_syn.columns if c in df_msrc.columns and not c.startswith("s_")]
        syn_sub = df_syn[cols].copy()
        syn_sub["is_synthetic"] = 1
        msrc_sub = df_msrc[cols].copy()
        # Offset extent IDs in MSRC to prevent ID collision
        max_syn_ext = int(df_syn["extent"].max()) + 1
        msrc_sub["extent"] = msrc_sub["extent"] + max_syn_ext
        msrc_sub["is_synthetic"] = 0
        combined = pd.concat([syn_sub, msrc_sub], ignore_index=True)
        return combined

    train_comb = concat_and_align(syn_splits.train_df, msrc_splits.train_df)
    cal_a_comb = concat_and_align(syn_splits.cal_a_df, msrc_splits.cal_a_df)
    cal_b_comb = concat_and_align(syn_splits.cal_b_df, msrc_splits.cal_b_df)
    test_comb = concat_and_align(syn_splits.test_df, msrc_splits.test_df)

    # Re-scale features strictly on combined train set
    scaler = RobustScaler()
    scaler.fit(train_comb[FEATURE_COLUMNS])

    scaled_cols = [f"s_{col}" for col in FEATURE_COLUMNS]
    for df in [train_comb, cal_a_comb, cal_b_comb, test_comb]:
        scaled_vals = scaler.transform(df[FEATURE_COLUMNS])
        for idx, scol in enumerate(scaled_cols):
            df[scol] = scaled_vals[:, idx]

    hybrid_splits = DatasetSplits(
        train_df=train_comb,
        cal_a_df=cal_a_comb,
        cal_b_df=cal_b_comb,
        test_df=test_comb,
        scaler=scaler,
        feature_cols=scaled_cols,
        target_col="Y",
        metadata={
            "dataset_mode": "hybrid",
            "volume": volume,
            "seed": seed,
            "train_rows": len(train_comb),
            "test_rows": len(test_comb),
        },
    )
    meta = {
        "dataset_mode": "hybrid",
        "volume": volume,
        "n_extents": syn_meta["n_extents"] + msrc_meta["n_extents"],
        "extent_size_bytes": int(config["data"]["extent_size_bytes"]),
    }
    return hybrid_splits, meta


def get_dataset_splits(
    mode: str,
    config: Dict[str, Any],
    seed: int = 42,
    volume: str = "hm_0",
) -> Tuple[DatasetSplits, Dict[str, Any]]:
    """Unified entry point to fetch splits for any of the 3 dataset modes."""
    mode = mode.lower().strip()
    if mode == "synthetic":
        return load_synthetic_feature_splits(config, seed=seed)
    elif mode == "msrc":
        return load_msrc_feature_splits(config, volume=volume)
    elif mode == "hybrid":
        return load_hybrid_feature_splits(config, seed=seed, volume=volume)
    else:
        raise ValueError(f"Unknown dataset mode '{mode}'. Must be 'synthetic', 'msrc', or 'hybrid'.")
