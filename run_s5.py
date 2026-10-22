"""
S5 Runner: Conformal Calibration, LTT Gate, Defer Branch, Validity Tests.

Runs:
1. Certifiability check (n_cal_epochs, min certifiable alpha).
2. CQR calibration on Cal-A (coverage analysis only).
3. LTT gate certification on Cal-B with Hoeffding-Bentkus bounds.
4. Defer branch demonstration.
5. Online tracker simulation.
6. Print summary and save to results/s5_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from src.data.synthetic import generate_workload
from src.data.preprocess import (
    build_dense_feature_table,
    create_purged_time_splits,
)
from src.simulator.cost_model import CostModel, CostModelConfig
from src.models.quantile_xgb import train_quantile_models
from src.models.prob_model import train_prob_model, predict_prob
from src.calibration.conformal import calibrate_cqr, evaluate_cqr_coverage
from src.calibration.ltt_gate import (
    run_ltt_certification,
    format_ltt_summary,
)
from src.calibration.defer import DeferConfig, DeferTracker
from src.calibration.online_tracker import OnlineTracker, TrackerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S5 Calibration & LTT Gate")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("S5: Conformal Calibration, LTT Gate, Defer Branch")
    print("=" * 70)

    # 1. Generate workload & preprocess
    print("\n[1/6] Generating workload and preprocessing ...")
    df_raw, meta = generate_workload(config, seed=config["seed"], workload_type="bursty")
    h = config["data"]["horizon_h"]
    feature_df = build_dense_feature_table(
        df_raw, meta["n_windows"], meta["n_extents"], h=h
    )
    splits = create_purged_time_splits(feature_df, meta["n_windows"], config)

    # Train quantile models (reuse from S4 — in practice we'd load saved models)
    print("\n[2/6] Training quantile models (10 levels) ...")
    quantile_levels = config["models"]["quantile_levels"]
    xgb_params = config["models"]["xgboost"]
    X_train = splits.train_df[splits.feature_cols].values
    y_train = splits.train_df[splits.target_col].values

    suite = train_quantile_models(
        X_train, y_train,
        quantile_levels=quantile_levels,
        xgb_params=xgb_params,
        verbose=True,
    )

    # Cost model
    cm_cfg = CostModelConfig(
        l_hdd_ms=config["simulator"]["hdd"]["mean_service_ms"],
        l_ssd_ms=config["simulator"]["ssd"]["mean_service_ms"],
        dirty_fraction=config["cost_model"]["dirty_fraction"],
        min_median_tau=5.0,
    )
    cm = CostModel(cm_cfg)
    sample_q50s = splits.train_df["ewma_5"].sample(
        n=min(500, len(splits.train_df)), random_state=42
    ).tolist()
    cm.check_and_rescale_tau_distribution(sample_q50s)

    # 3. CQR on Cal-A
    print("\n[3/6] CQR coverage analysis on Cal-A ...")
    alpha_cov = config["calibration"]["alpha_cov"]
    X_cal_a = splits.cal_a_df[splits.feature_cols].values
    y_cal_a = splits.cal_a_df[splits.target_col].values
    preds_cal_a = suite.predict_sorted(X_cal_a)

    cqr_result = calibrate_cqr(
        q_lo_cal=preds_cal_a[0.10],
        y_cal=y_cal_a,
        alpha_cov=alpha_cov,
    )
    print(f"  CQR offset Q: {cqr_result.Q_offset:.4f}")
    print(f"  Cal-A coverage P(Y >= Y_lo): {cqr_result.cal_coverage:.4f}")

    # CQR on Test
    X_test = splits.test_df[splits.feature_cols].values
    y_test = splits.test_df[splits.target_col].values
    preds_test = suite.predict_sorted(X_test)

    cqr_test_metrics = evaluate_cqr_coverage(
        preds_test[0.10], y_test, cqr_result.Q_offset
    )
    print(f"  Test coverage: {cqr_test_metrics['coverage']:.4f}")
    print(f"  Test nonzero Y_lo fraction: {cqr_test_metrics['nonzero_fraction']:.4f}")

    # 4. LTT Gate on Cal-B
    print("\n[4/6] LTT Gate certification on Cal-B ...")
    alpha_fmr = config["calibration"]["alpha_fmr"]
    delta_ltt = config["calibration"]["delta_ltt"]
    gate_levels = config["models"]["gate_quantile_levels"]
    delta_range = tuple(config["calibration"]["delta_range"])
    delta_steps = config["calibration"]["delta_steps"]
    n_shards = config["calibration"]["n_extent_shards"]

    X_cal_b = splits.cal_b_df[splits.feature_cols].values
    y_cal_b = splits.cal_b_df[splits.target_col].values
    preds_cal_b = suite.predict_sorted(X_cal_b)

    # Per-sample tau values
    tau_cal_b = np.array([cm.compute_tau(None) for _ in range(len(y_cal_b))])

    # Epoch and extent ids
    epoch_ids_cal_b = splits.cal_b_df["window"].values // h
    extent_ids_cal_b = splits.cal_b_df["extent"].values.astype(int)

    ltt_result = run_ltt_certification(
        quantile_preds={q: preds_cal_b[q] for q in gate_levels if q in preds_cal_b},
        tau_values=tau_cal_b,
        Y_actual=y_cal_b,
        epoch_ids=epoch_ids_cal_b,
        cost_model=cm,
        gate_quantile_levels=gate_levels,
        alpha_fmr=alpha_fmr,
        delta_ltt=delta_ltt,
        delta_range=delta_range,
        delta_steps=delta_steps,
        n_extent_shards=n_shards,
        extent_ids=extent_ids_cal_b,
        verbose=True,
    )
    print(f"\n  {format_ltt_summary(ltt_result)}")

    # 5. Defer branch demo
    print("\n[5/6] Defer branch demonstration ...")
    # Train prob model for defer
    preds_train = suite.predict_sorted(X_train)
    median_tau = cm.compute_tau(None)

    prob_result = train_prob_model(
        q10_train=preds_train[0.10],
        q50_train=preds_train[0.50],
        q90_train=preds_train[0.90],
        y_train=y_train,
        tau=median_tau,
        q10_cal_a=preds_cal_a[0.10],
        q50_cal_a=preds_cal_a[0.50],
        q90_cal_a=preds_cal_a[0.90],
        y_cal_a=y_cal_a,
        verbose=False,
    )

    # Demo on test set first epoch
    p_hat_test = predict_prob(
        prob_result,
        q10=preds_test[0.10],
        q50=preds_test[0.50],
        q90=preds_test[0.90],
    )

    defer_cfg = DeferConfig(
        p_defer=config["defer"]["p_defer"],
        max_defer_epochs=config["defer"]["max_defer_epochs"],
    )
    tracker = DeferTracker(defer_cfg)

    # Use best gate cell (or fallback)
    best_q = ltt_result.best_cell.q if ltt_result.best_cell else 0.10
    best_delta = ltt_result.best_cell.delta if ltt_result.best_cell else 0.0

    test_Y_lo = preds_test.get(best_q, preds_test[0.10])
    test_tau_thresholds = np.full(len(y_test), median_tau * (1 + best_delta))
    test_epoch_ids = splits.test_df["window"].values // h
    test_extent_ids = splits.test_df["extent"].values.astype(int)

    # Run for first 3 epochs
    unique_test_epochs = np.unique(test_epoch_ids)[:3]
    for ep in unique_test_epochs:
        ep_mask = test_epoch_ids == ep
        decisions = tracker.decide(
            test_extent_ids[ep_mask],
            test_Y_lo[ep_mask],
            test_tau_thresholds[ep_mask],
            p_hat_test[ep_mask],
            epoch=int(ep),
        )

    defer_summary = tracker.get_summary()
    print(f"  Defer summary (first 3 test epochs): {defer_summary}")

    # 6. Online tracker simulation
    print("\n[6/6] Online tracker simulation ...")
    online_cfg = TrackerConfig(
        eta=config["online_tracker"]["eta"],
        delta_min=config["online_tracker"]["delta_min"],
        delta_max=config["online_tracker"]["delta_max"],
        h=h,
    )
    initial_delta = best_delta
    online = OnlineTracker(
        config=online_cfg,
        initial_delta=initial_delta,
        alpha_fmr=alpha_fmr,
    )

    # Simulate with synthetic epoch FMRs from Cal-B result
    if ltt_result.best_cell and len(ltt_result.best_cell.fdp_values) > 0:
        epoch_fmrs = ltt_result.best_cell.fdp_values
    else:
        epoch_fmrs = np.random.default_rng(42).beta(2, 18, size=50)

    trajectory = online.simulate_trajectory(
        epoch_fmrs,
        feedback_delay=1,  # 1-epoch delay
    )
    online_summary = online.get_summary()
    print(f"  Initial delta: {initial_delta:.4f}")
    print(f"  Final delta: {online_summary['current_delta']:.4f}")
    print(f"  Updates: {online_summary['n_updates']}")
    print(f"  Mean FMR: {online_summary.get('mean_fmr', 'N/A')}")
    print(f"  FMR violations: {online_summary.get('fmr_violations', 'N/A')}")

    # Save results
    summary = {
        "cqr": {
            "Q_offset": cqr_result.Q_offset,
            "alpha_cov": cqr_result.alpha_cov,
            "cal_coverage": cqr_result.cal_coverage,
            "test_coverage": cqr_test_metrics["coverage"],
        },
        "ltt": {
            "certifiability": ltt_result.certifiability_report,
            "n_cal_epochs": ltt_result.n_cal_epochs,
            "n_cal_units": ltt_result.n_cal_units,
            "K": ltt_result.K,
            "min_certifiable_alpha": ltt_result.min_certifiable_alpha,
            "certified_cells": len(ltt_result.certified_cells),
            "total_cells": len(ltt_result.grid),
            "best_cell": {
                "q": ltt_result.best_cell.q if ltt_result.best_cell else None,
                "delta": ltt_result.best_cell.delta if ltt_result.best_cell else None,
                "mean_fdp": ltt_result.best_cell.mean_fdp if ltt_result.best_cell else None,
                "certified": ltt_result.best_cell.certified if ltt_result.best_cell else False,
                "net_benefit": ltt_result.best_cell.net_benefit if ltt_result.best_cell else None,
            },
        },
        "defer": defer_summary,
        "online_tracker": online_summary,
    }

    out_file = results_dir / "s5_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nS5 Summary written to {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
