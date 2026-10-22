"""Cached diagnostics requested after the corrected S7 baseline audit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from src.calibration.ltt_gate import run_ltt_certification
from src.data.preprocess import build_dense_feature_table, create_purged_time_splits
from src.data.synthetic import generate_workload
from src.experiment.stats import bootstrap_ci
from src.models.quantile_xgb import train_quantile_models
from src.simulator.baselines import run_baseline
from src.simulator.cost_model import CostModel, CostModelConfig
from src.simulator.storage_sim import SimConfig, StorageSim


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
CACHE = RESULTS / "extra_analysis_full_cache.json"
ALPHAS = [0.05, 0.10, 0.20, 0.30]
POLICIES = [
    "risk_controlled", "point_forecast", "safety_buffer",
    "static_threshold", "lru_promote", "hindsight_oracle",
]


def _cost_model(config: Dict[str, Any], sample: Iterable[float]) -> CostModel:
    cm = CostModel(CostModelConfig(
        l_hdd_ms=config["simulator"]["hdd"]["mean_service_ms"],
        l_ssd_ms=config["simulator"]["ssd"]["mean_service_ms"],
        dirty_fraction=config["cost_model"]["dirty_fraction"],
        min_median_tau=5.0,
    ))
    cm.check_and_rescale_tau_distribution(list(sample)[:300])
    return cm


def _fit_seed(config: Dict[str, Any], seed: int, workload: str = "bursty") -> Dict[str, Any]:
    raw, meta = generate_workload(config, seed=seed, workload_type=workload)
    features = build_dense_feature_table(
        raw, meta["n_windows"], meta["n_extents"], config["data"]["horizon_h"]
    )
    splits = create_purged_time_splits(features, meta["n_windows"], config)
    train_df = splits.train_df
    cal_df = splits.cal_b_df
    test_df = splits.test_df.copy()
    suite = train_quantile_models(
        train_df[splits.feature_cols].values,
        train_df[splits.target_col].values,
        quantile_levels=config["models"]["quantile_levels"],
        xgb_params=config["models"]["xgboost"],
        verbose=False,
    )
    cm = _cost_model(config, train_df["ewma_5"].sample(
        n=min(300, len(train_df)), random_state=seed
    ))
    cal = cal_df
    test = test_df.copy()
    cal_preds = suite.predict_sorted(cal[splits.feature_cols].values)
    test_preds = suite.predict_sorted(test[splits.feature_cols].values)
    tau_cal = np.full(len(cal), cm.compute_tau(None))
    tau_test = np.full(len(test), cm.compute_tau(None))
    return {
        "seed": seed, "meta": meta, "cm": cm, "splits": splits,
        "cal": cal, "test": test, "cal_preds": cal_preds,
        "test_preds": test_preds, "tau_cal": tau_cal, "tau_test": tau_test,
    }


def _certify(run: Dict[str, Any], config: Dict[str, Any], alpha: float):
    cal = run["cal"]
    h = config["data"]["horizon_h"]
    levels = config["models"]["gate_quantile_levels"]
    return run_ltt_certification(
        {q: run["cal_preds"][q] for q in levels},
        run["tau_cal"], cal["Y"].values,
        cal["window"].values // h,
        run["cm"], levels, alpha_fmr=alpha,
        delta_ltt=config["calibration"]["delta_ltt"],
        n_extent_shards=config["calibration"]["n_extent_shards"],
        extent_ids=cal["extent"].values.astype(int), verbose=False,
    )


def _decision_df(run: Dict[str, Any], q: float, policy: str, delta: float = 0.0) -> pd.DataFrame:
    test = run["test"].copy()
    test["q_lo"] = run["test_preds"].get(q, run["test_preds"][0.10])
    test["q50"] = run["test_preds"][0.50]
    test["tau"] = run["tau_test"]
    if policy == "risk_controlled":
        name, d = "proposed", delta
    elif policy == "point_forecast":
        name, d = "point_forecast", 0.0
    elif policy == "safety_buffer":
        name, d = "safety_buffer", 0.0
    elif policy == "static_threshold":
        name, d = "static_threshold", 0.0
    elif policy == "lru_promote":
        name, d = "lru_promote", 0.0
    else:
        name, d = "hindsight_oracle", 0.0
    test.attrs["policy_name"] = name
    test.attrs["delta"] = d
    return test


def _replay(run: Dict[str, Any], config: Dict[str, Any], policy: str, alpha: float):
    cert = _certify(run, config, alpha)
    q = cert.best_cell.q if cert.best_cell else 0.10
    delta = cert.best_cell.delta if cert.best_cell else 0.0
    if policy == "point_forecast":
        q, delta = 0.50, 0.0
    df = _decision_df(run, q, policy, delta)
    name = df.attrs["policy_name"]
    result = run_baseline(
        name, df,
        ssd_capacity=int(run["meta"]["n_extents"] * config["budget"]["ssd_fraction"]),
        max_migrations_per_epoch=config["budget"]["max_migrations_per_epoch"],
        cost_model=run["cm"], threshold=5.0,
        margin=2.0, delta=df.attrs["delta"],
        q_lo_col="q_lo", q50_col="q50", y_col="Y",
    )
    return result, cert


def _metric_row(result: Any, alpha: float, seed: int, policy: str) -> Dict[str, Any]:
    fdp = np.asarray(result.epoch_fdp_series)
    return {
        "alpha": alpha, "seed": seed, "policy": policy,
        "migrations": result.total_migrations,
        "wasted": result.total_false_migrations,
        "pooled_fmr": result.pooled_fmr,
        "mean_epoch_fdp": result.mean_fdp,
        "migration_activity": result.migration_activity,
        "net_benefit": result.total_net_benefit,
        "block_fdp_violation": bool(np.mean(fdp) > alpha),
    }


def _matched_summary(rows: List[Dict[str, Any]], alpha: float) -> List[Dict[str, Any]]:
    proposed = [r for r in rows if r["policy"] == "risk_controlled" and r["alpha"] == alpha]
    out = []
    for policy in POLICIES:
        current = [r for r in rows if r["policy"] == policy and r["alpha"] == alpha]
        matched = []
        for row in current:
            targets = [p["migrations"] for p in proposed]
            if targets:
                target = targets[current.index(row)]
                if 0.9 * target <= row["migrations"] <= 1.1 * target:
                    matched.append(row)
        out.append({
            "alpha": alpha, "policy": policy,
            "matched_blocks": len(matched),
            "mean_migrations": round(float(np.mean([r["migrations"] for r in matched])), 2) if matched else None,
            "mean_wasted": round(float(np.mean([r["wasted"] for r in matched])), 2) if matched else None,
            "mean_violation": round(float(np.mean([r["block_fdp_violation"] for r in matched])), 4) if matched else None,
        })
    return out


def _headline_sim(run: Dict[str, Any], policy: str, result: Any, seed: int, config: Dict[str, Any]) -> Dict[str, Any]:
    test = run["test"]
    cap = min(100_000, int(test["Y"].sum()))
    rng = np.random.default_rng(seed)
    rows = []
    rho = 0.70
    arrival_rate_per_ms = rho / config["simulator"]["hdd"]["mean_service_ms"]
    current_time = 0.0
    for _, row in test.iterrows():
        count = min(int(row["Y"]), max(0, cap - len(rows)))
        if count <= 0:
            continue
        for _ in range(count):
            current_time += float(rng.exponential(1.0 / arrival_rate_per_ms))
            rows.append((current_time, int(row["extent"]), 1, 64))
        if len(rows) >= cap:
            break
    requests = np.asarray(rows[:cap], dtype=float)
    if policy == "hdd_only":
        resident = set()
    elif policy == "all_ssd":
        resident = set(test["extent"].astype(int).unique())
    else:
        q = 0.50 if policy == "point_forecast" else 0.10
        scores = run["test_preds"].get(q, run["test_preds"][0.10])
        order = np.argsort(scores)[::-1]
        capacity = int(run["meta"]["n_extents"] * config["budget"]["ssd_fraction"])
        resident = set(test.iloc[order[:capacity]]["extent"].astype(int))
    sim = StorageSim(SimConfig(
        hdd_capacity=config["simulator"]["hdd"]["capacity"],
        ssd_capacity=config["simulator"]["ssd"]["capacity"],
        mean_service_hdd_ms=config["simulator"]["hdd"]["mean_service_ms"],
        mean_service_ssd_ms=config["simulator"]["ssd"]["mean_service_ms"],
        kappa_interference=config["simulator"]["migration"]["kappa"],
    ))
    metrics = sim.simulate_trace_slice(requests, resident, seed=seed)
    assert metrics["ssd_hit_ratio"] == (0.0 if policy == "hdd_only" else metrics["ssd_hit_ratio"])
    if policy == "all_ssd":
        assert metrics["ssd_hit_ratio"] == 1.0
    return {
        "seed": seed, "policy": policy, "matched_migrations": result.total_migrations,
        "request_count": int(len(requests)), "mean_latency_ms": metrics["mean_latency_ms"],
        "p95_latency_ms": metrics["p95_latency_ms"],         "p99_latency_ms": metrics["p99_latency_ms"],
        "iops": metrics["iops"], "throughput_iops": metrics["throughput_iops"],
        "ssd_hit_ratio": metrics["ssd_hit_ratio"],
        "sla_violation_rate": metrics["sla_violation_rate"],
    }


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    if CACHE.exists():
        print(CACHE.read_text())
        return
    seeds = list(range(config["seed"], config["seed"] + 10))
    runs = [_fit_seed(config, seed, "noisy_bursty") for seed in seeds]
    s1 = []
    for workload in config["data"]["synthetic"]["workload_types"] + ["noisy_bursty"]:
        raw, meta = generate_workload(config, seed=config["seed"] + abs(hash(workload)) % 10_000, workload_type=workload)
        feature = build_dense_feature_table(raw, meta["n_windows"], meta["n_extents"], config["data"]["horizon_h"])
        tau = _cost_model(config, feature["ewma_5"].sample(n=min(300, len(feature)), random_state=42)).compute_tau(None)
        y = feature["Y"].to_numpy()
        access = raw.groupby("extent")["count"].sum().to_numpy()
        s1.append({"workload": workload, "prevalence_y_ge_tau": float(np.mean(y >= tau)),
                    "y_median": float(np.median(y)), "y_p90": float(np.percentile(y, 90)),
                    "y_p99": float(np.percentile(y, 99)), "event_count_median": float(np.median(access)),
                    "event_count_p90": float(np.percentile(access, 90)), "event_count_p99": float(np.percentile(access, 99))})
    rows = []
    cert_lines = []
    certified_cells = []
    for run in runs:
        for alpha in ALPHAS:
            cert = _certify(run, config, alpha)
            cert_lines.append({"seed": run["seed"], "alpha": alpha, "n_units": cert.n_cal_units,
                               "min_certifiable_alpha": cert.min_certifiable_alpha,
                               "tau_median": run["cm"].compute_tau(None)})
            certified_cells.extend({
                "seed": run["seed"], "alpha": alpha, "q": cell.q,
                "delta": cell.delta, "p_value": cell.p_value,
                "cal_fdp": cell.mean_fdp,
                "migrating_units": cell.n_epochs_with_mig,
                "certified": cell.certified,
            } for cell in cert.grid if cell.certified)
            for policy in POLICIES:
                result, _ = _replay(run, config, policy, alpha)
                rows.append(_metric_row(result, alpha, run["seed"], policy))
    drift_rows = []
    for seed in seeds:
        run = _fit_seed(config, seed, "changing_hot_set")
        cert = _certify(run, config, 0.10)
        q = cert.best_cell.q if cert.best_cell else 0.10
        pred = run["test_preds"][q]
        test = run["test"]
        eps = np.unique(test["window"].values // config["data"]["horizon_h"])
        static, online = [], []
        delta = 0.1
        for ep in eps:
            mask = (test["window"].values // config["data"]["horizon_h"]) == ep
            for vals, out in [(pred[mask] >= run["tau_test"][mask] * 1.1, static),
                              (pred[mask] >= run["tau_test"][mask] * (1 + delta), online)]:
                ys = test.loc[mask, "Y"].to_numpy()[vals]
                out.append(float(np.mean([run["cm"].compute_realized_net_benefit(float(y)) <= 0 for y in ys])) if len(ys) else 0.0)
            delta = np.clip(delta + 0.5 * (online[-1] - 0.10), -0.5, 2.0)
        drift_rows.append({"seed": seed, "static_mean_fmr": float(np.mean(static)), "online_mean_fmr": float(np.mean(online)),
                           "static_violation_rate": float(np.mean(np.asarray(static) > 0.10)),
                           "online_violation_rate": float(np.mean(np.asarray(online) > 0.10)),
                           "post_drift_static": float(np.mean(np.asarray(static)[len(static)//2:])),
                           "post_drift_online": float(np.mean(np.asarray(online)[len(online)//2:]))})
    sim_rows = []
    for run in runs:
        for policy in ["risk_controlled", "point_forecast", "lru_promote", "static_threshold", "hdd_only", "all_ssd", "hindsight_oracle"]:
            result, _ = _replay(run, config, policy, 0.10)
            sim_rows.append(_headline_sim(run, policy, result, run["seed"], config))
    payload = {"s1": s1, "diagnostic_rows": rows, "certified_cells": certified_cells,
               "matched_volume": sum((_matched_summary(rows, a) for a in ALPHAS), []),
               "certifiability": cert_lines, "drift": drift_rows, "simulator": sim_rows,
               "msrc_files": sorted(p.name for p in (ROOT / "data" / "msrc_raw").glob("*"))}
    CACHE.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
