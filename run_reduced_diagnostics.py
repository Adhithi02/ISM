"""Precheck and reduced diagnostic jobs; never invokes S7."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from src.calibration.ltt_gate import run_ltt_certification
from src.data.msrc_loader import load_msrc_volumes
from src.data.preprocess import build_dense_feature_table, create_purged_time_splits
from src.data.synthetic import generate_workload
from src.models.quantile_xgb import train_quantile_models
from src.simulator.baselines import run_baseline
from src.simulator.cost_model import CostModel, CostModelConfig


ROOT = Path(__file__).parent
ALPHAS = [0.05, 0.10, 0.20, 0.30]


def load_config() -> Dict[str, Any]:
    return yaml.safe_load((ROOT / "config" / "default.yaml").read_text())


def make_cost(config: Dict[str, Any], scale: float = 1.0, minimum: float = 5.0) -> CostModel:
    return CostModel(CostModelConfig(
        l_hdd_ms=config["simulator"]["hdd"]["mean_service_ms"],
        l_ssd_ms=config["simulator"]["ssd"]["mean_service_ms"],
        dirty_fraction=config["cost_model"]["dirty_fraction"],
        c_mig_scale=scale,
        min_median_tau=minimum,
    ))


def precheck() -> bool:
    config = load_config()
    config = copy.deepcopy(config)
    config["data"]["synthetic"]["n_windows"] = 500
    config["data"]["synthetic"]["n_extents"] = 500
    config["budget"]["ssd_fraction"] = 0.50
    config["budget"]["max_migrations_per_epoch"] = 500
    config["calibration"]["n_extent_shards"] = 32
    raw, meta = generate_workload(config, seed=config["seed"], workload_type="noisy_bursty")
    features = build_dense_feature_table(raw, 500, 500, h=5)
    cm = make_cost(config, scale=0.15, minimum=0.5)
    cm.check_and_rescale_tau_distribution([0.0] * 300)
    tau = np.full(len(features), cm.compute_tau(None))
    y = features["Y"].to_numpy()
    q50 = np.maximum(0.0, y + np.random.default_rng(7).normal(0.0, 2.0, len(y)))
    q10 = np.maximum(0.0, q50 - 3.0)
    decision = features.copy()
    decision["q_lo"] = q10
    decision["q50"] = q50
    decision["tau"] = tau
    oracle = run_baseline(
        "hindsight_oracle", decision, 250, 500, cm, q_lo_col="q_lo", q50_col="q50"
    )
    policy_rows = []
    for policy in ["hindsight_oracle", "point_forecast", "static_threshold", "lru_promote"]:
        result = run_baseline(
            policy, decision, 250, 500, cm, threshold=1.0,
            q_lo_col="q_lo", q50_col="q50"
        )
        policy_rows.append((policy, result.total_migrations, result.migration_activity))
    print("NOISY_BURSTY_PRECHECK")
    print(f"oracle_migrations={oracle.total_migrations}")
    print(f"oracle_activity={oracle.migration_activity:.4f}")
    print(f"prevalence_y_ge_tau={np.mean(y >= cm.compute_tau(None)):.4f}")
    print(f"y_percentiles=median:{np.median(y):.2f},p90:{np.percentile(y,90):.2f},p99:{np.percentile(y,99):.2f}")
    print(f"tau_median={cm.compute_tau(None):.4f}")
    for policy, migrations, activity in policy_rows:
        print(f"{policy}:migrations={migrations},activity={activity:.4f}")
    cal = features[(features["window"] >= 205) & (features["window"] < 355)].copy()
    cal_preds = {q: (q10 if q <= 0.10 else q50) for q in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]}
    cal_ids = cal.index.to_numpy()
    for alpha in ALPHAS:
        result = run_ltt_certification(
            {q: pred[cal_ids] for q, pred in cal_preds.items()},
            tau[cal_ids], cal["Y"].to_numpy(), cal["window"].to_numpy() // 5,
            cm, list(cal_preds), alpha_fmr=alpha, delta_ltt=0.10,
            n_extent_shards=32, extent_ids=cal["extent"].to_numpy(), verbose=False,
        )
        cells = [c for c in result.grid if c.certified]
        best = max(cells, key=lambda c: c.net_benefit) if cells else result.best_cell
        print(
            f"alpha={alpha:.2f}:cells={len(cells)},"
            f"best=({best.q:.2f},{best.delta:.2f})" if best else
            f"alpha={alpha:.2f}:cells=0,best=none"
        )
        if best:
            print(f"alpha={alpha:.2f}:best_p={best.p_value:.6f},best_fdp={best.mean_fdp:.4f},units={best.n_epochs_with_mig}")
    passed = oracle.total_migrations >= 200 and all(activity >= 0.30 for _, _, activity in policy_rows)
    print(f"PASS={passed}")
    return passed


def fit_and_log(config: Dict[str, Any], seed: int, workload: str, output: Path) -> None:
    raw, meta = generate_workload(config, seed=seed, workload_type=workload)
    features = build_dense_feature_table(raw, meta["n_windows"], meta["n_extents"], config["data"]["horizon_h"])
    splits = create_purged_time_splits(features, meta["n_windows"], config)
    suite = train_quantile_models(
        splits.train_df[splits.feature_cols].values,
        splits.train_df[splits.target_col].values,
        quantile_levels=config["models"]["quantile_levels"],
        xgb_params=config["models"]["xgboost"],
        verbose=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "seed": seed, "workload": workload, "rows": len(raw),
        "windows": meta["n_windows"], "extents": meta["n_extents"],
        "train_rows": len(splits.train_df), "cal_b_rows": len(splits.cal_b_df),
        "test_rows": len(splits.test_df), "models": len(suite.models),
    }) + "\n")


def synthetic_job() -> None:
    config = load_config()
    config["data"]["synthetic"]["workload_types"] = ["noisy_bursty"]
    for seed in range(config["seed"], config["seed"] + 5):
        print(f"seed={seed}", flush=True)
        fit_and_log(config, seed, "noisy_bursty", ROOT / "results" / "logs" / "synthetic_last.json")
    print("synthetic_complete", flush=True)


def msrc_job() -> None:
    config = load_config()
    config["data"]["msrc"]["volumes"] = ["hm_0", "prn_0"]
    loaded, missing = load_msrc_volumes(config, stop_if_missing=False)
    summary = {}
    for volume, frame in loaded.items():
        summary[volume] = {
            "present": frame is not None,
            "rows": int(len(frame)) if frame is not None else 0,
            "extents": int(frame["extent"].nunique()) if frame is not None else 0,
        }
    print(json.dumps({"volumes": summary, "missing": missing}), flush=True)
    (ROOT / "results" / "logs" / "msrc_summary.json").write_text(json.dumps(summary, indent=2))
    print("msrc_complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["precheck", "synthetic", "msrc"])
    args = parser.parse_args()
    if args.mode == "precheck":
        if not precheck():
            raise SystemExit("Noisy-bursty precheck failed; long jobs not started.")
    elif args.mode == "synthetic":
        synthetic_job()
    else:
        msrc_job()


if __name__ == "__main__":
    main()
