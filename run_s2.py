"""
S2 runner: Preprocessing, Feature Engineering, and 4-Way Splitting with Purge Gaps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from src.data.synthetic import generate_workload
from src.data.preprocess import build_dense_feature_table, create_purged_time_splits, FEATURE_COLUMNS


def run_s2_for_workload(config: Dict[str, Any], workload_type: str = "bursty") -> Dict[str, Any]:
    seed = config["seed"]
    h = config["data"]["horizon_h"]
    df_raw, meta = generate_workload(config, seed=seed, workload_type=workload_type)

    feature_df = build_dense_feature_table(
        df_raw,
        n_windows=meta["n_windows"],
        n_extents=meta["n_extents"],
        h=h,
    )

    splits = create_purged_time_splits(feature_df, meta["n_windows"], config)

    # Summary dictionary
    summary = {
        "workload_type": workload_type,
        "n_raw_events": len(df_raw),
        "n_windows": meta["n_windows"],
        "n_extents": meta["n_extents"],
        "horizon_h": h,
        "n_decision_rows": len(feature_df),
        "splits": splits.metadata,
        "features": {
            "raw_count": len(FEATURE_COLUMNS),
            "columns": FEATURE_COLUMNS,
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S2 Preprocessing & Splitting")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--workload", default="bursty")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("S2: Preprocessing, Features & 4-Way Splitting (Purge Gaps)")
    print("=" * 60)

    summary = run_s2_for_workload(config, workload_type=args.workload)
    meta = summary["splits"]
    bounds = meta["boundaries"]
    rows = meta["rows"]

    print(f"\nWorkload Type: {args.workload}")
    print(f"Total Decision Epoch Rows: {summary['n_decision_rows']:,}")
    print(f"Features: {summary['features']['raw_count']} columns")

    print("\n[4-Way Time-Ordered Partitioning]")
    print(f"  Train : Windows {bounds['train'][0]:>4} .. {bounds['train'][1]:>4} | Rows: {rows['train']:>6,}")
    print(f"    --- Purge Gap: {meta['purge_gap_windows']} windows (w={bounds['train'][1]}..{bounds['cal_a'][0]-1}) ---")
    print(f"  Cal-A : Windows {bounds['cal_a'][0]:>4} .. {bounds['cal_a'][1]:>4} | Rows: {rows['cal_a']:>6,}")
    print(f"    --- Purge Gap: {meta['purge_gap_windows']} windows (w={bounds['cal_a'][1]}..{bounds['cal_b'][0]-1}) ---")
    print(f"  Cal-B : Windows {bounds['cal_b'][0]:>4} .. {bounds['cal_b'][1]:>4} | Rows: {rows['cal_b']:>6,}")
    print(f"    --- Purge Gap: {meta['purge_gap_windows']} windows (w={bounds['cal_b'][1]}..{bounds['test'][0]-1}) ---")
    print(f"  Test  : Windows {bounds['test'][0]:>4} .. {bounds['test'][1]:>4} | Rows: {rows['test']:>6,}")

    out_file = results_dir / "s2_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {out_file}")


if __name__ == "__main__":
    main()
