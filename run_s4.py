"""
S4 Runner: Quantile Models, Probability Model, First Results.

Runs:
1. Train 10 quantile XGBoost models (Q = {0.05..0.50, 0.90}).
2. Evaluate pinball loss, MAE, coverage per quantile on Test set.
3. Report post-hoc crossing rate.
4. Compute [q10, q90] interval coverage and width.
5. Train isotonic probability model P_hat(Y >= tau | x).
6. Print results table and save to results/s4_summary.json.
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
    FEATURE_COLUMNS,
)
from src.simulator.cost_model import CostModel, CostModelConfig
from src.models.quantile_xgb import (
    train_quantile_models,
    evaluate_quantile_models,
    compute_interval_metrics,
)
from src.models.prob_model import train_prob_model, predict_prob


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S4 Quantile & Probability Models")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("S4: Quantile Models, Probability Model, First Results")
    print("=" * 70)

    # 1. Generate workload & preprocess
    print("\n[1/5] Generating synthetic workload and building features ...")
    df_raw, meta = generate_workload(config, seed=config["seed"], workload_type="bursty")
    h = config["data"]["horizon_h"]
    feature_df = build_dense_feature_table(
        df_raw, meta["n_windows"], meta["n_extents"], h=h
    )
    splits = create_purged_time_splits(feature_df, meta["n_windows"], config)

    print(f"  Split sizes: Train={len(splits.train_df):,}, Cal-A={len(splits.cal_a_df):,}, "
          f"Cal-B={len(splits.cal_b_df):,}, Test={len(splits.test_df):,}")

    # 2. Train quantile models
    print("\n[2/5] Training quantile XGBoost models ...")
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
    print(f"  Trained {len(suite.models)} quantile models: {sorted(suite.models.keys())}")

    # 3. Evaluate on Test set
    print("\n[3/5] Evaluating quantile models on Test set ...")
    X_test = splits.test_df[splits.feature_cols].values
    y_test = splits.test_df[splits.target_col].values

    eval_df = evaluate_quantile_models(suite, X_test, y_test)

    print(f"\n  Post-hoc crossing rate: {suite.crossing_rate:.4f}")
    print(f"\n  {'Quantile':>10} {'Pinball':>10} {'MAE':>10} {'P(Y<=q)':>10}")
    print("  " + "-" * 44)
    for _, row in eval_df.iterrows():
        print(f"  {row['quantile']:>10.2f} {row['pinball_loss']:>10.4f} "
              f"{row['mae']:>10.4f} {row['coverage_below']:>10.4f}")

    # 4. Interval metrics [q10, q90]
    print("\n[4/5] Prediction interval [q10, q90] metrics ...")
    interval_metrics = compute_interval_metrics(
        suite, X_test, y_test, q_lo=0.10, q_hi=0.90
    )
    print(f"  Coverage: {interval_metrics['coverage']:.4f} "
          f"(nominal: {interval_metrics['nominal_coverage']:.2f})")
    print(f"  Mean width: {interval_metrics['mean_width']:.2f}, "
          f"Median width: {interval_metrics['median_width']:.2f}")

    # 5. Train isotonic probability model
    print("\n[5/5] Training isotonic probability model P_hat(Y >= tau) ...")

    # Compute median tau for the threshold
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
    tau_stats = cm.check_and_rescale_tau_distribution(sample_q50s)
    median_tau = tau_stats["final_median_tau"]
    print(f"  Using median tau = {median_tau:.2f} as threshold")

    # Get quantile predictions for prob model features
    preds_train = suite.predict_sorted(X_train)
    X_cal_a = splits.cal_a_df[splits.feature_cols].values
    y_cal_a = splits.cal_a_df[splits.target_col].values
    preds_cal_a = suite.predict_sorted(X_cal_a)

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
        verbose=True,
    )

    # Evaluate on Test set
    preds_test_sorted = suite.predict_sorted(X_test)
    p_hat_test = predict_prob(
        prob_result,
        q10=preds_test_sorted[0.10],
        q50=preds_test_sorted[0.50],
        q90=preds_test_sorted[0.90],
    )
    # Calibration on test
    z_test = (y_test >= median_tau).astype(np.int32)
    brier_test = float(np.mean((p_hat_test - z_test) ** 2))
    prevalence_test = float(z_test.mean())
    print(f"  Brier score (Test): {brier_test:.4f}")
    print(f"  Prevalence Y>=tau (Test): {prevalence_test:.3f}")
    print(f"  Mean P_hat (Test): {p_hat_test.mean():.3f}, "
          f"P_hat >= 0.5 fraction: {(p_hat_test >= 0.5).mean():.3f}")

    # Save results
    summary = {
        "quantile_levels": quantile_levels,
        "n_models": len(suite.models),
        "crossing_rate": round(suite.crossing_rate, 4),
        "eval_table": eval_df.to_dict("records"),
        "interval_metrics": interval_metrics,
        "prob_model": {
            "tau_threshold": median_tau,
            "cal_metrics": prob_result.cal_metrics,
            "brier_test": round(brier_test, 4),
            "prevalence_test": round(prevalence_test, 4),
        },
        "split_sizes": {
            "train": len(splits.train_df),
            "cal_a": len(splits.cal_a_df),
            "cal_b": len(splits.cal_b_df),
            "test": len(splits.test_df),
        },
    }

    out_file = results_dir / "s4_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nS4 Summary written to {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
