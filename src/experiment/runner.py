"""
Experiment orchestrator for ISM.

Runs the full pipeline for one (workload, seed) configuration:
  1. Generate workload, preprocess, split
  2. Train quantile models
  3. CQR calibration, LTT certification
  4. Budget selection, online tracking
  5. Evaluate on test set
  6. Return all metrics

Uses joblib for parallel execution across seeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.synthetic import generate_workload
from src.data.preprocess import build_dense_feature_table, create_purged_time_splits
from src.simulator.cost_model import CostModel, CostModelConfig
from src.models.quantile_xgb import train_quantile_models
from src.calibration.ltt_gate import run_ltt_certification
from src.experiment.metrics import compute_decision_metrics, DecisionMetrics


@dataclass
class RunResult:
    """Result of a single experiment run."""
    seed: int
    workload_type: str
    decision_metrics: DecisionMetrics
    ltt_certified: bool
    best_q: float
    best_delta: float
    crossing_rate: float
    n_cal_units: int


def run_single_experiment(
    config: Dict[str, Any],
    seed: int,
    workload_type: str = "bursty",
    alpha_fmr: Optional[float] = None,
    ssd_fraction: Optional[float] = None,
    cost_multiplier: float = 1.0,
    policy: str = "risk_controlled",
    verbose: bool = False,
) -> RunResult:
    """
    Run one complete experiment pipeline for a single seed.

    Parameters
    ----------
    config : master config dict
    seed : random seed
    workload_type : workload type string
    alpha_fmr : override alpha_FMR (None = use config default)
    ssd_fraction : override SSD fraction
    cost_multiplier : multiplier for C_mig
    policy : "risk_controlled" or "point_forecast"
    verbose : print progress
    """
    if policy not in {"risk_controlled", "point_forecast"}:
        raise ValueError(f"Unknown policy: {policy}")
    if alpha_fmr is None:
        alpha_fmr = config["calibration"]["alpha_fmr"]
    if ssd_fraction is None:
        ssd_fraction = config["budget"]["ssd_fraction"]

    h = config["data"]["horizon_h"]

    # 1. Generate & preprocess
    df_raw, meta = generate_workload(config, seed=seed, workload_type=workload_type)
    feature_df = build_dense_feature_table(
        df_raw, meta["n_windows"], meta["n_extents"], h=h
    )
    splits = create_purged_time_splits(feature_df, meta["n_windows"], config)

    # 2. Train quantile models (reduced n_estimators for speed in ablations)
    X_train = splits.train_df[splits.feature_cols].values
    y_train = splits.train_df[splits.target_col].values

    suite = train_quantile_models(
        X_train, y_train,
        quantile_levels=config["models"]["quantile_levels"],
        xgb_params={**config["models"]["xgboost"], "n_estimators": 150},
        verbose=False,
    )

    # 3. Cost model
    cm_cfg = CostModelConfig(
        l_hdd_ms=config["simulator"]["hdd"]["mean_service_ms"],
        l_ssd_ms=config["simulator"]["ssd"]["mean_service_ms"],
        dirty_fraction=config["cost_model"]["dirty_fraction"],
        min_median_tau=5.0,
        c_mig_scale=cost_multiplier,
        extent_size_mib=config["cost_model"].get("extent_size_mib", 1.0),
        kappa_interference=config["cost_model"].get("kappa_interference", 0.0),
        measured_utilization=config["cost_model"].get("measured_utilization", 0.0),
        ssd_wear_cost_per_mib=config["cost_model"].get("ssd_wear_cost_per_mib", 0.0),
    )
    cm = CostModel(cm_cfg)
    sample_q50s = splits.train_df["ewma_5"].sample(
        n=min(300, len(splits.train_df)), random_state=seed
    ).tolist()
    cm.check_and_rescale_tau_distribution(sample_q50s)

    # 4. LTT on Cal-B
    X_cal_b = splits.cal_b_df[splits.feature_cols].values
    y_cal_b = splits.cal_b_df[splits.target_col].values
    preds_cal_b = suite.predict_sorted(X_cal_b)

    gate_levels = config["models"]["gate_quantile_levels"]
    tau_cal_b = np.full(len(y_cal_b), cm.compute_tau(None))
    epoch_ids = splits.cal_b_df["window"].values // h
    extent_ids = splits.cal_b_df["extent"].values.astype(int)

    ltt_result = run_ltt_certification(
        quantile_preds={q: preds_cal_b[q] for q in gate_levels if q in preds_cal_b},
        tau_values=tau_cal_b,
        Y_actual=y_cal_b,
        epoch_ids=epoch_ids,
        cost_model=cm,
        gate_quantile_levels=gate_levels,
        alpha_fmr=alpha_fmr,
        delta_ltt=config["calibration"]["delta_ltt"],
        n_extent_shards=config["calibration"]["n_extent_shards"],
        extent_ids=extent_ids,
        verbose=False,
    )

    if policy == "point_forecast":
        best_q = 0.50
        best_delta = 0.0
    else:
        best_q = 0.10
        best_delta = 0.0
        if ltt_result.best_cell is not None:
            q_cell = ltt_result.best_cell.q
            if q_cell == q_cell:
                best_q = float(q_cell)
            best_delta = float(ltt_result.best_cell.delta)

    # 5. Evaluate on Test
    X_test = splits.test_df[splits.feature_cols].values
    y_test = splits.test_df[splits.target_col].values
    preds_test = suite.predict_sorted(X_test)

    test_Y_lo = preds_test.get(best_q, preds_test[0.10])
    test_tau = np.full(len(y_test), cm.compute_tau(None))
    test_threshold = test_tau * (1 + best_delta)
    gate_mask = test_Y_lo >= test_threshold

    test_epoch_ids = splits.test_df["window"].values // h
    unique_epochs = np.unique(test_epoch_ids)

    epoch_fdps = []
    epoch_migs = []
    epoch_wasted = []
    epoch_nbs = []

    for ep in unique_epochs:
        ep_mask = test_epoch_ids == ep
        ep_gate = gate_mask[ep_mask]
        n_mig = int(ep_gate.sum())

        if n_mig == 0:
            epoch_fdps.append(0.0)
            epoch_migs.append(0)
            epoch_wasted.append(0)
            epoch_nbs.append(0.0)
            continue

        ep_y = y_test[ep_mask][ep_gate]
        wasted = 0
        nb = 0.0
        for y_val in ep_y:
            this_nb = cm.compute_realized_net_benefit(float(y_val))
            nb += this_nb
            if this_nb <= 0:
                wasted += 1

        epoch_fdps.append(wasted / n_mig)
        epoch_migs.append(n_mig)
        epoch_wasted.append(wasted)
        epoch_nbs.append(nb)

    dm = compute_decision_metrics(
        np.array(epoch_fdps),
        np.array(epoch_migs),
        np.array(epoch_wasted),
        np.array(epoch_nbs),
    )

    return RunResult(
        seed=seed,
        workload_type=workload_type,
        decision_metrics=dm,
        ltt_certified=(
            policy == "risk_controlled"
            and ltt_result.best_cell.certified
            if ltt_result.best_cell
            else False
        ),
        best_q=best_q,
        best_delta=best_delta,
        crossing_rate=suite.crossing_rate,
        n_cal_units=ltt_result.n_cal_units,
    )
