"""A1 diagnosis of the current LTT gate. Does not change certification logic."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from src.calibration.ltt_gate import run_ltt_certification
from src.data.preprocess import build_dense_feature_table, create_purged_time_splits
from src.data.synthetic import generate_workload
from src.models.quantile_xgb import train_quantile_models
from src.simulator.baselines import run_baseline
from src.simulator.cost_model import CostModel, CostModelConfig

ROOT = Path(__file__).parent


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    cfg = copy.deepcopy(config)
    cfg["data"]["synthetic"]["n_windows"] = min(1000, cfg["data"]["synthetic"]["n_windows"])
    seed = cfg["seed"]
    raw, meta = generate_workload(cfg, seed=seed, workload_type="noisy_bursty")
    features = build_dense_feature_table(
        raw, meta["n_windows"], meta["n_extents"], cfg["data"]["horizon_h"]
    )
    splits = create_purged_time_splits(features, meta["n_windows"], cfg)
    fast_levels = [0.05, 0.10, 0.20, 0.30, 0.50, 0.90]
    suite = train_quantile_models(
        splits.train_df[splits.feature_cols].values,
        splits.train_df[splits.target_col].values,
        quantile_levels=fast_levels,
        xgb_params={**cfg["models"]["xgboost"], "n_estimators": 100, "subsample": 0.5},
        verbose=False,
    )
    cm = CostModel(CostModelConfig(
        l_hdd_ms=cfg["simulator"]["hdd"]["mean_service_ms"],
        l_ssd_ms=cfg["simulator"]["ssd"]["mean_service_ms"],
        dirty_fraction=cfg["cost_model"]["dirty_fraction"],
        extent_size_mib=cfg["cost_model"]["extent_size_mib"],
        kappa_interference=cfg["cost_model"]["kappa_interference"],
        measured_utilization=cfg["cost_model"]["measured_utilization"],
        ssd_wear_cost_per_mib=cfg["cost_model"]["ssd_wear_cost_per_mib"],
    ))
    cm.check_and_rescale_tau_distribution([0.0] * min(300, len(splits.train_df)))
    cal = splits.cal_b_df
    cal_preds = suite.predict_sorted(cal[splits.feature_cols].values)
    tau_cal = np.full(len(cal), cm.compute_tau(None))
    gate_qs = [q for q in fast_levels if q <= 0.50]
    test = splits.test_df.copy()
    preds = suite.predict_sorted(test[splits.feature_cols].values)
    test["q50"] = preds[0.50]
    test["q_lo"] = preds[0.10]
    test.attrs["extent_size_mib"] = cfg["cost_model"]["extent_size_mib"]
    capacity = max(1, int(meta["n_extents"] * cfg["budget"]["ssd_fraction"]))
    budget = int(cfg["budget"]["max_migrations_per_epoch"])

    tables = {}
    for alpha in cfg["experiment"]["alpha_sweep"]:
        ltt = run_ltt_certification(
            {q: cal_preds[q] for q in gate_qs},
            tau_cal,
            cal["Y"].to_numpy(),
            cal["window"].to_numpy() // cfg["data"]["horizon_h"],
            cm,
            gate_qs,
            alpha_fmr=alpha,
            delta_ltt=cfg["calibration"]["delta_ltt"],
            n_extent_shards=cfg["calibration"]["n_extent_shards"],
            extent_ids=cal["extent"].to_numpy(),
            verbose=False,
        )
        certified = [c for c in ltt.grid if c.certified]
        best = ltt.best_cell
        smallest_p = min((c.p_value for c in ltt.grid), default=None)
        chosen = None if best is None else {
            "q": best.q,
            "delta": best.delta,
            "p_value": best.p_value,
            "certified": best.certified,
            "mean_fdp": best.mean_fdp,
            "pooled_fmr": best.pooled_fmr,
            "net_benefit": best.net_benefit,
            "migration_activity": best.migration_activity,
            "total_migrations": best.total_migrations,
            "n_epochs_with_mig": best.n_epochs_with_mig,
        }
        eval_q10 = run_baseline(
            "proposed", test, capacity, budget, cm,
            delta=float(best.delta if best is not None else 0.0),
            extent_size_mib=cfg["cost_model"]["extent_size_mib"],
        )
        test_matched = test.copy()
        q_use = 0.10 if best is None else best.q
        test_matched["q_lo"] = preds.get(q_use, preds[0.10])
        eval_matched = run_baseline(
            "proposed", test_matched, capacity, budget, cm,
            delta=float(best.delta if best is not None else 0.0),
            extent_size_mib=cfg["cost_model"]["extent_size_mib"],
        )
        tables[str(alpha)] = {
            "n_cal_epochs": ltt.n_cal_epochs,
            "n_cal_units": ltt.n_cal_units,
            "n_certified_cells": len(certified),
            "migrating_units_chosen_cell": None if best is None else best.n_epochs_with_mig,
            "smallest_p_value": smallest_p,
            "chosen": chosen,
            "eval_always_q10_migrations": eval_q10.total_migrations,
            "eval_matched_q_migrations": eval_matched.total_migrations,
            "certifiability_report": ltt.certifiability_report,
        }

    point = run_baseline(
        "point_forecast", test, capacity, budget, cm,
        extent_size_mib=cfg["cost_model"]["extent_size_mib"],
    )
    out = {
        "seed": seed,
        "scenario": "noisy_bursty",
        "point_forecast_migrations": point.total_migrations,
        "tables": tables,
        "dominant_cause": (
            "LTT certifies a (q, delta) cell on Cal-B, but the test policy always "
            "applies delta to q10 instead of the certified quantile, so the deployed "
            "gate is far more conservative than the certified cell; Bonferroni over "
            "every quantile and mean FDP on inactive units further favours quiet cells."
        ),
    }
    path = ROOT / "results" / "a1_diagnosis.json"
    path.write_text(json.dumps(out, indent=2))
    lines = [
        "# A1 Gate diagnosis (current LTT, noisy_bursty seed 42)",
        "",
        out["dominant_cause"],
        "",
    ]
    for alpha in cfg["experiment"]["alpha_sweep"]:
        t = tables[str(alpha)]
        c = t["chosen"] or {}
        lines += [
            f"## alpha = {alpha:.2f}",
            "",
            "| Quantity | Value |",
            "|---|---|",
            f"| Calibration units (epoch x shard) | {t['n_cal_units']} |",
            f"| Calibration epochs | {t['n_cal_epochs']} |",
            f"| Certified cells | {t['n_certified_cells']} |",
            f"| Migrating units in chosen cell | {t['migrating_units_chosen_cell']} |",
            f"| Smallest p-value reached | {t['smallest_p_value']} |",
            f"| Chosen (q, delta) | ({c.get('q')}, {c.get('delta')}) |",
            f"| Chosen p-value | {c.get('p_value')} |",
            f"| Chosen certified | {c.get('certified')} |",
            f"| Chosen Cal-B activity | {c.get('migration_activity')} |",
            f"| Chosen Cal-B migrations | {c.get('total_migrations')} |",
            f"| Test migrations if always q10 | {t['eval_always_q10_migrations']} |",
            f"| Test migrations if matched q | {t['eval_matched_q_migrations']} |",
            "",
        ]
    lines += [
        f"Point-forecast test migrations: {point.total_migrations}",
        "",
        "Dominant cause: certified (q, delta) is not the score applied at test time "
        "(always q10), so the deployed policy is nearly silent.",
        "",
    ]
    (ROOT / "results" / "a1_diagnosis.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
