"""
Algorithm Benchmarking and Real-Time Storage Simulator Translation Engine for ISM.

Evaluates 5 Quantile Regression Algorithms:
- XGBoost
- LightGBM
- CatBoost
- HistGradientBoosting (Sklearn)
- Linear Quantile Regression

Across 3 Dataset Regimes:
- Synthetic-only (Bursty / Decoy / Drift)
- Hybrid (Synthetic pretraining + Real Enterprise Trace adaptation)
- MSRC-only (Pure Microsoft Research Cambridge Enterprise Traces)

Quantifies:
1. Forecast accuracy & uncertainty calibration (Pinball loss, MAE, Coverage, Width, Crossing rate, Train time)
2. Risk-controlled policy & safety metrics (FDP, Pooled FMR, Net benefit, Recall, Migration activity)
3. Discrete-event queue simulation for real-time translation (SimPy latency, p95, p99, IOPS, SSD hit ratio, SLA violations)
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

from src.calibration.ltt_gate import run_ltt_certification
from src.data.dataset_modes import get_dataset_splits
from src.models.model_registry import (
    SUPPORTED_ALGORITHMS,
    QuantileModelSuite,
    compute_interval_metrics,
    evaluate_quantile_models,
    train_quantile_model_suite,
)
from src.models.prob_model import predict_family2, train_family2_cal_a
from src.simulator.baselines import run_baseline
from src.simulator.cost_model import CostModel, CostModelConfig
from src.simulator.eviction import ValueEviction
from src.simulator.fast_replay import ReplayMetrics, run_fast_replay
from src.simulator.storage_sim import SimConfig, StorageSim

QUANTILE_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.90]
GATE_QS = [0.05, 0.10, 0.20, 0.30, 0.50]
SIM_MAX_REQUESTS = 20_000


def make_cost_model(config: Dict[str, Any], extent_size_mib: float = 16.0) -> CostModel:
    return CostModel(CostModelConfig(
        l_hdd_ms=config["simulator"]["hdd"]["mean_service_ms"],
        l_ssd_ms=config["simulator"]["ssd"]["mean_service_ms"],
        dirty_fraction=config["cost_model"]["dirty_fraction"],
        extent_size_mib=extent_size_mib,
        kappa_interference=config["cost_model"]["kappa_interference"],
        measured_utilization=config["cost_model"]["measured_utilization"],
        ssd_wear_cost_per_mib=config["cost_model"]["ssd_wear_cost_per_mib"],
    ))


def generate_sim_requests(test_df: pd.DataFrame, window_sec: float, cap: int, seed: int = 42) -> np.ndarray:
    """Generate timestamped I/O request stream from test partition for SimPy queue replay."""
    rng = np.random.default_rng(seed)
    rows: List[List[float]] = []
    t0 = float(test_df["window"].min()) * window_sec * 1000.0
    for _, r in test_df.iterrows():
        n = int(max(0, round(float(r["Y"]))))
        if n <= 0:
            continue
        base = float(r["window"]) * window_sec * 1000.0 - t0
        for _ in range(n):
            rows.append([base + rng.uniform(0, window_sec * 1000.0), float(r["extent"]), 1.0, 64.0])
            if len(rows) >= cap:
                break
        if len(rows) >= cap:
            break
    if not rows:
        return np.zeros((0, 4), dtype=float)
    arr = np.asarray(rows, dtype=float)
    return arr[np.argsort(arr[:, 0])]


def scale_service_for_target_rho(sim: StorageSim, requests: np.ndarray, target_rho: float = 0.70) -> float:
    if len(requests) < 2:
        return sim.cfg.mean_service_hdd_ms
    duration_ms = max(1.0, float(requests[-1, 0] - requests[0, 0]))
    arrival_per_ms = len(requests) / duration_ms
    service_ms = target_rho / max(arrival_per_ms, 1e-9)
    sim.cfg.mean_service_hdd_ms = float(np.clip(service_ms, 0.05, 50.0))
    return sim.cfg.mean_service_hdd_ms


def run_benchmark_for_setting(
    algorithm: str,
    dataset_mode: str,
    config: Dict[str, Any],
    seed: int = 42,
    volume: str = "hm_0",
    alpha_fmr: float = 0.10,
) -> Dict[str, Any]:
    """Execute complete training, calibration, replay, and SimPy simulation for one (algorithm, dataset) pair."""
    # 1. Load dataset splits
    splits, meta = get_dataset_splits(dataset_mode, config, seed=seed, volume=volume)
    extent_size_mib = float(meta.get("extent_size_bytes", 16 * 1024 * 1024)) / (1024 * 1024)

    # 2. Train Quantile Model Suite
    X_train = splits.train_df[splits.feature_cols].values
    y_train = splits.train_df[splits.target_col].values

    suite = train_quantile_model_suite(
        algorithm=algorithm,
        X_train=X_train,
        y_train=y_train,
        quantile_levels=QUANTILE_LEVELS,
        random_state=seed,
        verbose=False,
    )

    # 3. Forecast Accuracy & Uncertainty Metrics on Test Split
    X_test = splits.test_df[splits.feature_cols].values
    y_test = splits.test_df[splits.target_col].values
    test_eval_df = evaluate_quantile_models(suite, X_test, y_test)
    int_metrics = compute_interval_metrics(suite, X_test, y_test, q_lo=0.10, q_hi=0.90)

    pinball_avg = float(test_eval_df["pinball_loss"].mean())
    mae_median = float(test_eval_df[test_eval_df["quantile"] == 0.50]["mae"].values[0]) if 0.50 in test_eval_df["quantile"].values else 0.0

    # 4. Calibration & Gate Certification on Cal-B
    cm = make_cost_model(config, extent_size_mib=extent_size_mib)
    sample_q50s = [0.0] * min(300, max(1, len(splits.train_df)))
    cm.check_and_rescale_tau_distribution(sample_q50s)
    tau_value = cm.compute_tau(None)

    # Family 2 Calibrated Probability
    cal_a_preds = suite.predict_sorted(splits.cal_a_df[splits.feature_cols].values)
    tau_cal_a = np.full(len(splits.cal_a_df), tau_value)
    fam2_model = train_family2_cal_a(
        cal_a_preds,
        splits.cal_a_df["Y"].to_numpy(),
        tau_cal_a,
        verbose=False,
    )

    # Cal-B predictions
    cal_b_preds = suite.predict_sorted(splits.cal_b_df[splits.feature_cols].values)
    tau_cal_b = np.full(len(splits.cal_b_df), tau_value)
    p_hat_cal_b = predict_family2(fam2_model, cal_b_preds, tau_cal_b)

    epoch_cal_b = splits.cal_b_df["window"].values // config["data"]["horizon_h"]
    extent_cal_b = splits.cal_b_df["extent"].values

    ltt = run_ltt_certification(
        quantile_preds={q: cal_b_preds[q] for q in GATE_QS},
        tau_values=tau_cal_b,
        Y_actual=splits.cal_b_df["Y"].values,
        epoch_ids=epoch_cal_b,
        cost_model=cm,
        gate_quantile_levels=GATE_QS,
        alpha_fmr=alpha_fmr,
        delta_ltt=config["calibration"]["delta_ltt"],
        n_extent_shards=config["calibration"]["n_extent_shards"],
        extent_ids=extent_cal_b,
        p_hat=p_hat_cal_b,
        verbose=False,
    )

    best_cell = ltt.best_cell
    cert_status = "CERTIFIED" if (best_cell and best_cell.certified) else "UNCERTIFIED"

    # 5. Fast Trace Replay on Test Split
    test_preds = suite.predict_sorted(X_test)
    test_df = splits.test_df.copy()
    for q, arr in test_preds.items():
        test_df[f"q{int(round(q * 100)):02d}"] = arr
    test_df["q50"] = test_preds[0.50]
    test_df["q_lo"] = test_preds[0.10]
    p_hat_test = predict_family2(fam2_model, test_preds, np.full(len(test_df), tau_value))
    test_df["p_hat"] = p_hat_test

    ssd_capacity = max(2, int(meta["n_extents"] * config["budget"]["ssd_fraction"]))
    max_migs = int(config["budget"]["max_migrations_per_epoch"])

    # Oracle benchmark for recall calculation
    oracle_replay = run_baseline(
        "hindsight_oracle",
        test_df,
        ssd_capacity=ssd_capacity,
        max_migrations_per_epoch=max_migs,
        cost_model=cm,
        extent_size_mib=extent_size_mib,
    )
    oracle_log = set(oracle_replay.decision_log)

    # Proposed Risk-Controlled Policy execution
    if best_cell and best_cell.family == "probability":
        p_star = float(best_cell.p_star if best_cell.p_star is not None else 0.5)
        gate_fn = lambda c, tau, ps=p_star: c["p_hat"] >= ps
        replay = run_fast_replay(
            test_df, ssd_capacity, max_migs, cm, ValueEviction(), gate_fn=gate_fn, extent_size_mib=extent_size_mib
        )
    elif best_cell:
        q_sel = 0.10 if (best_cell.q != best_cell.q) else float(best_cell.q)
        delta_sel = float(best_cell.delta)
        q_col = f"q{int(round(q_sel * 100)):02d}"
        gate_fn = lambda c, tau, d=delta_sel, col=q_col: c[col] >= tau * (1.0 + d)
        replay = run_fast_replay(
            test_df, ssd_capacity, max_migs, cm, ValueEviction(), gate_fn=gate_fn, extent_size_mib=extent_size_mib
        )
    else:
        # Fallback conservative gate
        gate_fn = lambda c, tau: c["q_lo"] >= tau
        replay = run_fast_replay(
            test_df, ssd_capacity, max_migs, cm, ValueEviction(), gate_fn=gate_fn, extent_size_mib=extent_size_mib
        )

    recall = 0.0
    if oracle_log:
        overlap = set(replay.decision_log).intersection(oracle_log)
        recall = len(overlap) / len(oracle_log)

    # 6. Discrete-Event Simulator (SimPy) Real-Time Translation
    req_stream = generate_sim_requests(
        test_df,
        window_sec=float(config["data"]["window_sec"]),
        cap=SIM_MAX_REQUESTS,
        seed=seed,
    )
    ssd_resident_extents = {int(e) for _, e in replay.decision_log}

    sim = StorageSim(SimConfig(
        mean_service_hdd_ms=config["simulator"]["hdd"]["mean_service_ms"],
        mean_service_ssd_ms=config["simulator"]["ssd"]["mean_service_ms"],
        kappa_interference=config["cost_model"]["kappa_interference"],
        extent_size_mib=extent_size_mib,
    ))
    scale_service_for_target_rho(sim, req_stream, target_rho=0.70)
    sim_metrics = sim.simulate_trace_slice(req_stream, ssd_resident_extents, seed=seed)

    return {
        "algorithm": algorithm,
        "dataset_mode": dataset_mode,
        "training_time_s": suite.training_time_s,
        "crossing_rate": round(suite.crossing_rate, 4),
        "pinball_loss_avg": round(pinball_avg, 4),
        "mae_median": round(mae_median, 4),
        "interval_coverage_80": int_metrics["coverage"],
        "interval_mean_width": int_metrics["mean_width"],
        "brier_score": fam2_model.cal_metrics.get("brier", 0.0),
        "ece": fam2_model.cal_metrics.get("ece", 0.0),
        "ltt_certified": bool(best_cell.certified if best_cell else False),
        "ltt_status": cert_status,
        "chosen_family": best_cell.family if best_cell else "none",
        "chosen_param": (f"p*={best_cell.p_star}" if best_cell and best_cell.family == "probability" else f"q={best_cell.q}, delta={best_cell.delta}") if best_cell else "none",
        "migrations": replay.total_migrations,
        "wasted_migrations": replay.total_false_migrations,
        "mean_fdp": replay.mean_fdp,
        "pooled_fmr": replay.pooled_fmr,
        "migration_activity": replay.migration_activity,
        "net_benefit": replay.total_net_benefit,
        "recall_vs_oracle": round(recall, 4),
        "sim_mean_latency_ms": round(sim_metrics["mean_latency_ms"], 2),
        "sim_p95_latency_ms": round(sim_metrics["p95_latency_ms"], 2),
        "sim_p99_latency_ms": round(sim_metrics["p99_latency_ms"], 2),
        "sim_iops": round(sim_metrics["iops"], 1),
        "sim_ssd_hit_ratio": round(sim_metrics["ssd_hit_ratio"], 4),
        "sim_sla_violation_rate": round(sim_metrics["sla_violation_rate"], 4),
        "sim_hdd_utilization": round(sim_metrics["hdd_utilization"], 4),
    }


def run_full_algorithm_benchmark(
    config_path: str = "config/default.yaml",
    algorithms: Optional[List[str]] = None,
    dataset_modes: Optional[List[str]] = None,
    seed: int = 42,
    volume: str = "hm_0",
) -> List[Dict[str, Any]]:
    """Runs cross-product of all algorithms x all dataset configurations."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    algos = algorithms or SUPPORTED_ALGORITHMS
    modes = dataset_modes or ["synthetic", "hybrid", "msrc"]

    print("=" * 80)
    print("ISM ADVANCED ALGORITHM & DATASET BENCHMARK SUITE")
    print(f"Algorithms: {algos}")
    print(f"Dataset Modes: {modes}")
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    total_runs = len(algos) * len(modes)
    curr = 0

    for mode in modes:
        print(f"\n>>> Running Dataset Mode: {mode.upper()} <<<")
        for algo in algos:
            curr += 1
            print(f"[{curr}/{total_runs}] Evaluating {algo.upper()} on {mode} ... ", end="", flush=True)
            t_start = time.perf_counter()
            try:
                res = run_benchmark_for_setting(
                    algorithm=algo,
                    dataset_mode=mode,
                    config=config,
                    seed=seed,
                    volume=volume,
                )
                dur = time.perf_counter() - t_start
                print(f"Done ({dur:.1f}s) | Pinball={res['pinball_loss_avg']:.3f} | NetBen={res['net_benefit']:.1f} | p99={res['sim_p99_latency_ms']:.1f}ms")
                results.append(res)
            except Exception as e:
                print(f"FAILED: {e}")
                results.append({
                    "algorithm": algo,
                    "dataset_mode": mode,
                    "error": str(e),
                })

    return results
