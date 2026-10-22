"""
S3 runner: Storage Simulators, Eviction Policies, Cost Model & Baselines.

Runs:
1. State-dependent break-even tau distribution check and rescaling (median >= 5.0).
2. Plot tau distribution histogram -> plots/tau_distribution.png.
3. Fast replay baseline latency & net benefit comparison across all 9 policies.
4. Outputs summary table and results/s3_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data.synthetic import generate_workload
from src.data.preprocess import build_dense_feature_table
from src.simulator.cost_model import CostModel, CostModelConfig
from src.simulator.baselines import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S3 Simulator and Baselines")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("S3: Simulators, Cost Model, Eviction Policies & Baselines")
    print("=" * 60)

    # 1. Generate workload slice
    print("\n[1/3] Generating synthetic workload slice for S3 baseline run...")
    df_raw, meta = generate_workload(config, seed=config["seed"], workload_type="bursty")
    h = config["data"]["horizon_h"]
    feature_df = build_dense_feature_table(df_raw, meta["n_windows"], meta["n_extents"], h=h)

    # 2. Cost model and tau distribution
    print("\n[2/3] Evaluating tau distribution & rescaling...")
    cm_cfg = CostModelConfig(
        l_hdd_ms=config["simulator"]["hdd"]["mean_service_ms"],
        l_ssd_ms=config["simulator"]["ssd"]["mean_service_ms"],
        dirty_fraction=config["cost_model"]["dirty_fraction"],
        min_median_tau=5.0,
    )
    cm = CostModel(cm_cfg)

    # Sample evictee values (q50 proxy using rolling count)
    sample_q50s = feature_df["ewma_5"].sample(n=min(1000, len(feature_df)), random_state=42).tolist()
    tau_stats = cm.check_and_rescale_tau_distribution(sample_q50s)

    print(f"  Raw median tau   : {tau_stats['raw_median_tau']:.2f}")
    print(f"  Rescale factor   : {tau_stats['rescale_factor']:.3f} (applied to C_mig)")
    print(f"  Final median tau : {tau_stats['final_median_tau']:.2f} (Target >= 5.0)")
    print(f"  p10: {tau_stats['p10']:.2f}, p90: {tau_stats['p90']:.2f}")

    # Plot tau distribution
    taus = [cm.compute_tau(q) for q in sample_q50s]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(taus, bins=30, color="#2b5c8f", edgecolor="white", alpha=0.85)
    ax.axvline(tau_stats['final_median_tau'], color="#e040fb", linestyle="--", linewidth=2,
               label=f"Median tau = {tau_stats['final_median_tau']:.1f}")
    ax.set_title("Break-Even Migration Threshold tau(x) Distribution")
    ax.set_xlabel("tau (accesses in next h windows to break even)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    tau_plot_path = plots_dir / "tau_distribution.png"
    plt.savefig(tau_plot_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {tau_plot_path}")

    # 3. Baselines comparison using FastReplay
    print("\n[3/3] Running all 9 migration baselines on decision epochs...")
    ssd_fraction = config["budget"]["ssd_fraction"]
    ssd_capacity = int(meta["n_extents"] * ssd_fraction)
    max_migs = config["budget"]["max_migrations_per_epoch"]

    policies = [
        ("hdd_only", "None"),
        ("all_ssd", "None"),
        ("lru_promote", "LRU"),
        ("static_threshold", "LRU"),
        ("point_forecast", "LRU"),
        ("point_forecast_value_evict", "Value"),
        ("safety_buffer", "LRU"),
        ("proposed_lru_evict", "LRU"),
        ("proposed", "Value"),
        ("hindsight_oracle", "Value"),
    ]

    summary_records = []
    print(f"\n{'Policy':<28} {'Eviction':<8} {'Migs':>7} {'Activity':>9} {'Mean FDP':>9} {'Pooled FMR':>11} {'Net Ben (ms)':>13}")
    print("-" * 92)

    for pol, evict in policies:
        metrics = run_baseline(
            policy_name=pol,
            decision_df=feature_df,
            ssd_capacity=ssd_capacity,
            max_migrations_per_epoch=max_migs,
            cost_model=cm,
            q50_col="ewma_5",
            q_lo_col="count_lag1",
            y_col="Y",
        )
        rec = {
            "policy": pol,
            "eviction": evict,
            "total_migrations": metrics.total_migrations,
            "migration_activity": metrics.migration_activity,
            "mean_fdp": metrics.mean_fdp,
            "pooled_fmr": metrics.pooled_fmr,
            "total_net_benefit_ms": metrics.total_net_benefit,
        }
        summary_records.append(rec)
        print(f"{pol:<28} {evict:<8} {metrics.total_migrations:>7,} {metrics.migration_activity*100:>8.1f}% "
              f"{metrics.mean_fdp:>9.3f} {metrics.pooled_fmr:>11.3f} {metrics.total_net_benefit:>13,.1f}")

    out_file = results_dir / "s3_summary.json"
    with open(out_file, "w") as f:
        json.dump({
            "tau_stats": tau_stats,
            "ssd_capacity": ssd_capacity,
            "max_migrations_per_epoch": max_migs,
            "baselines": summary_records,
        }, f, indent=2)

    print(f"\nS3 Summary written to {out_file}")


if __name__ == "__main__":
    main()
