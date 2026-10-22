"""Cheap physical-cost regime gate for the final experiment configuration."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from src.data.preprocess import build_dense_feature_table
from src.data.synthetic import generate_workload
from src.simulator.baselines import run_baseline
from src.simulator.cost_model import CostModel, CostModelConfig

ROOT = Path(__file__).parent


def main() -> None:
    base = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    attempts = []
    for i, (rate, spike_rate) in enumerate(
        [(800.0, 0.80), (500.0, 0.50), (350.0, 0.35), (250.0, 0.25), (180.0, 0.18)],
        start=1,
    ):
        cfg = copy.deepcopy(base)
        cfg["data"]["synthetic"]["n_windows"] = 500
        cfg["data"]["synthetic"]["n_extents"] = 500
        cfg["data"]["synthetic"]["base_total_rate"] = rate
        cfg["data"]["synthetic"]["spikes_per_extent"]["noisy_bursty_rate"] = spike_rate
        cfg["budget"]["ssd_fraction"] = 0.20
        cfg["budget"]["max_migrations_per_epoch"] = 200
        raw, meta = generate_workload(cfg, seed=cfg["seed"], workload_type="noisy_bursty")
        features = build_dense_feature_table(raw, 500, 500, h=cfg["data"]["horizon_h"])
        cm = CostModel(CostModelConfig(
            l_hdd_ms=cfg["simulator"]["hdd"]["mean_service_ms"],
            l_ssd_ms=cfg["simulator"]["ssd"]["mean_service_ms"],
            dirty_fraction=cfg["cost_model"]["dirty_fraction"],
            extent_size_mib=cfg["cost_model"]["extent_size_mib"],
            kappa_interference=cfg["cost_model"]["kappa_interference"],
            measured_utilization=cfg["cost_model"]["measured_utilization"],
            ssd_wear_cost_per_mib=cfg["cost_model"]["ssd_wear_cost_per_mib"],
        ))
        stats = cm.check_and_rescale_tau_distribution([0.0] * 300)
        decision = features.copy()
        decision["q_lo"] = np.maximum(0.0, decision["Y"].to_numpy() - 3.0)
        decision["q50"] = decision["Y"].to_numpy()
        oracle = run_baseline(
            "hindsight_oracle", decision, 100, 200, cm,
            q_lo_col="q_lo", q50_col="q50",
            extent_size_mib=cfg["cost_model"]["extent_size_mib"],
        )
        rows = {}
        for policy in ("hindsight_oracle", "point_forecast", "static_threshold", "lru_promote", "proposed"):
            result = run_baseline(
                policy, decision, 100, 200, cm, threshold=5.0,
                q_lo_col="q_lo", q50_col="q50",
                extent_size_mib=cfg["cost_model"]["extent_size_mib"],
            )
            rows[policy] = {
                "migrations": result.total_migrations,
                "activity": result.migration_activity,
            }
        prevalence = float(np.mean(decision["Y"].to_numpy() >= cm.compute_tau(None)))
        row = {
            "attempt": i,
            "base_total_rate": rate,
            "spike_rate": spike_rate,
            "tau": cm.compute_tau(None),
            "tau_stats": stats,
            "prevalence": prevalence,
            "policies": rows,
        }
        attempts.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if (
            stats["final_median_tau"] >= 5.0
            and 0.05 <= prevalence <= 0.30
            and rows["hindsight_oracle"]["activity"] < 0.6
            and rows["hindsight_oracle"]["migrations"] >= 200
        ):
            print("PASS", flush=True)
            (ROOT / "results" / "regime_gate.json").write_text(json.dumps(row, indent=2))
            return
    (ROOT / "results" / "regime_gate.json").write_text(json.dumps({"status": "failed", "attempts": attempts}, indent=2))
    print("STOP: regime gate failed after five sparse-workload attempts", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
