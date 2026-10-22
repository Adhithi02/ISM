"""
S6 Runner: Budgeted Selection, Online Tracker Drift Test.

Runs:
1. Budget module demonstration with greedy knapsack.
2. Online tracker drift test: changing_hot_set workload,
   static gate vs online tracker FMR over time.
3. Save drift plot and results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.data.synthetic import generate_workload
from src.data.preprocess import build_dense_feature_table, create_purged_time_splits
from src.simulator.cost_model import CostModel, CostModelConfig
from src.models.quantile_xgb import train_quantile_models
from src.calibration.ltt_gate import run_ltt_certification
from src.calibration.online_tracker import OnlineTracker, TrackerConfig
from src.selection.budget import select_migrations_for_epoch
from src.viz.plots import plot_fmr_over_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S6 Budget & Online Tracker")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("S6: Budgeted Selection, Online Tracker, Drift Test")
    print("=" * 70)

    h = config["data"]["horizon_h"]

    # 1. Budget demonstration on bursty workload
    print("\n[1/3] Budget selection demonstration ...")
    df_raw, meta = generate_workload(config, seed=config["seed"], workload_type="bursty")
    feature_df = build_dense_feature_table(df_raw, meta["n_windows"], meta["n_extents"], h=h)
    splits = create_purged_time_splits(feature_df, meta["n_windows"], config)

    X_train = splits.train_df[splits.feature_cols].values
    y_train = splits.train_df[splits.target_col].values
    suite = train_quantile_models(
        X_train, y_train,
        quantile_levels=config["models"]["quantile_levels"],
        xgb_params=config["models"]["xgboost"],
        verbose=False,
    )

    cm_cfg = CostModelConfig(
        l_hdd_ms=config["simulator"]["hdd"]["mean_service_ms"],
        l_ssd_ms=config["simulator"]["ssd"]["mean_service_ms"],
        dirty_fraction=config["cost_model"]["dirty_fraction"],
        min_median_tau=5.0,
    )
    cm = CostModel(cm_cfg)
    cm.check_and_rescale_tau_distribution(
        splits.train_df["ewma_5"].sample(n=300, random_state=42).tolist()
    )

    # Demo on first test epoch
    X_test = splits.test_df[splits.feature_cols].values
    y_test = splits.test_df[splits.target_col].values
    preds_test = suite.predict_sorted(X_test)
    test_epoch_ids = splits.test_df["window"].values // h
    first_epoch = np.unique(test_epoch_ids)[0]
    ep_mask = test_epoch_ids == first_epoch

    tau_val = cm.compute_tau(None)
    gate_mask = preds_test[0.10][ep_mask] >= tau_val * 1.1

    budget_result = select_migrations_for_epoch(
        extent_ids=splits.test_df["extent"].values[ep_mask].astype(int),
        q50_preds=preds_test[0.50][ep_mask],
        tau_values=np.full(int(ep_mask.sum()), tau_val),
        gate_mask=gate_mask,
        max_migrations=config["budget"]["max_migrations_per_epoch"],
        ssd_available=int(meta["n_extents"] * config["budget"]["ssd_fraction"]),
    )
    print(f"  First test epoch: {len(budget_result.selected)} selected, "
          f"{len(budget_result.rejected)} rejected")
    print(f"  BW used: {budget_result.budget_bw_used}/{budget_result.budget_bw_limit}, "
          f"SSD used: {budget_result.budget_ssd_used}/{budget_result.budget_ssd_limit}")
    print(f"  Total benefit: {budget_result.total_benefit:.2f}")

    # 2. Drift test: changing_hot_set
    print("\n[2/3] Drift test: changing_hot_set workload ...")
    df_drift, meta_drift = generate_workload(
        config, seed=config["seed"], workload_type="changing_hot_set"
    )
    feat_drift = build_dense_feature_table(
        df_drift, meta_drift["n_windows"], meta_drift["n_extents"], h=h
    )
    splits_drift = create_purged_time_splits(feat_drift, meta_drift["n_windows"], config)

    X_train_d = splits_drift.train_df[splits_drift.feature_cols].values
    y_train_d = splits_drift.train_df[splits_drift.target_col].values
    suite_d = train_quantile_models(
        X_train_d, y_train_d,
        quantile_levels=config["models"]["quantile_levels"],
        xgb_params={**config["models"]["xgboost"], "n_estimators": 150},
        verbose=False,
    )

    X_test_d = splits_drift.test_df[splits_drift.feature_cols].values
    y_test_d = splits_drift.test_df[splits_drift.target_col].values
    preds_test_d = suite_d.predict_sorted(X_test_d)

    # Compute per-epoch FMR for static gate
    alpha_fmr = config["calibration"]["alpha_fmr"]
    test_epochs_d = splits_drift.test_df["window"].values // h
    unique_eps = np.unique(test_epochs_d)

    static_delta = 0.1  # fixed
    online_tracker = OnlineTracker(
        config=TrackerConfig(
            eta=config["online_tracker"]["eta"],
            delta_min=config["online_tracker"]["delta_min"],
            delta_max=config["online_tracker"]["delta_max"],
            h=h,
        ),
        initial_delta=static_delta,
        alpha_fmr=alpha_fmr,
    )

    static_fmrs = []
    online_fmrs = []
    epoch_list = []

    for ep in unique_eps:
        ep_mask = test_epochs_d == ep
        ep_y = y_test_d[ep_mask]
        ep_q10 = preds_test_d[0.10][ep_mask]

        # Static gate
        s_gate = ep_q10 >= tau_val * (1 + static_delta)
        s_mig = int(s_gate.sum())
        if s_mig > 0:
            s_wasted = sum(1 for y in ep_y[s_gate]
                          if cm.compute_realized_net_benefit(float(y)) <= 0)
            s_fmr = s_wasted / s_mig
        else:
            s_fmr = 0.0

        # Online gate
        cur_delta = online_tracker.current_delta
        o_gate = ep_q10 >= tau_val * (1 + cur_delta)
        o_mig = int(o_gate.sum())
        if o_mig > 0:
            o_wasted = sum(1 for y in ep_y[o_gate]
                          if cm.compute_realized_net_benefit(float(y)) <= 0)
            o_fmr = o_wasted / o_mig
        else:
            o_fmr = 0.0

        # Update tracker with delayed feedback (use current epoch FMR)
        online_tracker.update(int(ep), o_fmr)

        static_fmrs.append(s_fmr)
        online_fmrs.append(o_fmr)
        epoch_list.append(int(ep))

    static_arr = np.array(static_fmrs)
    online_arr = np.array(online_fmrs)
    epoch_arr = np.array(epoch_list)

    print(f"  Static gate: mean FMR={np.mean(static_arr):.4f}, "
          f"violations={int(np.sum(static_arr > alpha_fmr))}/{len(static_arr)}")
    print(f"  Online tracker: mean FMR={np.mean(online_arr):.4f}, "
          f"violations={int(np.sum(online_arr > alpha_fmr))}/{len(online_arr)}")

    # 3. Plot and save
    print("\n[3/3] Saving drift plot and results ...")
    # Smooth for readability (rolling mean window=5)
    window = min(5, len(static_arr))
    if window > 1:
        s_smooth = np.convolve(static_arr, np.ones(window)/window, mode="valid")
        o_smooth = np.convolve(online_arr, np.ones(window)/window, mode="valid")
        ep_smooth = epoch_arr[:len(s_smooth)]
    else:
        s_smooth, o_smooth, ep_smooth = static_arr, online_arr, epoch_arr

    plot_fmr_over_time(
        ep_smooth, s_smooth, o_smooth, alpha_fmr,
        save_path=plots_dir / "drift_fmr_static_vs_online.png",
    )
    print(f"  Saved: plots/drift_fmr_static_vs_online.png")

    summary = {
        "budget_demo": {
            "selected": len(budget_result.selected),
            "rejected": len(budget_result.rejected),
            "total_benefit": budget_result.total_benefit,
        },
        "drift_test": {
            "workload": "changing_hot_set",
            "n_epochs": len(epoch_list),
            "static_mean_fmr": round(float(np.mean(static_arr)), 4),
            "online_mean_fmr": round(float(np.mean(online_arr)), 4),
            "static_violations": int(np.sum(static_arr > alpha_fmr)),
            "online_violations": int(np.sum(online_arr > alpha_fmr)),
            "tracker_summary": online_tracker.get_summary(),
        },
    }

    out_file = results_dir / "s6_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nS6 Summary written to {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
