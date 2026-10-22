"""MSRC hm_0 / prn_0 evaluation with the same cost model and 16 MiB extents."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.preprocess import FEATURE_COLUMNS, build_dense_feature_table, create_purged_time_splits
from src.experiment.full_job import (
    FAST_LEVELS,
    GATE_QS,
    attach_preds,
    make_cost_model,
    metrics_from_replay,
)
from src.calibration.ltt_gate import run_ltt_certification
from src.models.prob_model import predict_family2, train_family2_cal_a
from src.models.quantile_xgb import train_quantile_models
from src.simulator.baselines import run_baseline

ROOT = Path(__file__).parent


def window_aggregate(df: pd.DataFrame, window_sec: float) -> pd.DataFrame:
    t0 = float(df["timestamp_s"].min())
    w = ((df["timestamp_s"] - t0) / window_sec).astype(np.int32)
    out = df.copy()
    out["window"] = w
    grp = out.groupby(["window", "extent"], sort=False)
    agg = grp.agg(
        count=("extent", "size"),
        read_count=("is_read", "sum"),
        mean_size_bytes=("size_bytes", "mean"),
    ).reset_index()
    agg["write_count"] = agg["count"] - agg["read_count"]
    agg["timestamp_start_s"] = t0 + agg["window"] * window_sec
    return agg


def remap_extents(df: pd.DataFrame) -> pd.DataFrame:
    uniq = np.sort(df["extent"].unique())
    mapping = {int(e): i for i, e in enumerate(uniq)}
    out = df.copy()
    out["extent"] = out["extent"].map(mapping).astype(np.int32)
    return out, len(uniq)


def inspect_parquet(path: Path) -> dict:
    df = pd.read_parquet(path)
    return {
        "path": str(path),
        "columns": list(df.columns),
        "rows": int(len(df)),
        "first_five": df.head(5).to_dict(orient="records"),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }


def eval_volume(name: str, parquet: Path, config: dict) -> dict:
    raw = pd.read_parquet(parquet)
    header = inspect_parquet(parquet)
    window_sec = float(config["data"]["window_sec"])
    agg = window_aggregate(raw, window_sec)
    agg, n_ext = remap_extents(agg)
    n_windows = int(agg["window"].max()) + 1
    notes = []
    # Cap windows for tractable training; document any change.
    max_windows = 1500
    if n_windows > max_windows:
        notes.append(f"sliced to first {max_windows} of {n_windows} windows for tractable training")
        agg = agg[agg["window"] < max_windows].copy()
        n_windows = max_windows
    features = build_dense_feature_table(
        agg, n_windows, n_ext, config["data"]["horizon_h"],
        extent_size_bytes=int(config["data"]["extent_size_bytes"]),
    )
    splits = create_purged_time_splits(features, n_windows, config)
    cm = make_cost_model(config)
    cm.check_and_rescale_tau_distribution([0.0] * min(300, max(1, len(splits.train_df))))
    tau = cm.compute_tau(None)
    y = features["Y"].to_numpy() if len(features) else np.array([])
    prevalence = float(np.mean(y >= tau)) if len(y) else 0.0
    if prevalence < 0.02 or prevalence > 0.40:
        notes.append(
            f"prevalence Y>=tau is {prevalence:.4f} with unchanged tau={tau:.4f}; "
            "did not lower tau. Evaluation still runs on this volume."
        )
    if len(splits.train_df) < 50 or len(splits.cal_b_df) < 30 or len(splits.test_df) < 30:
        return {
            "volume": name, "header": header, "n_windows": n_windows, "n_extents": n_ext,
            "tau": tau, "prevalence": prevalence, "notes": notes + ["insufficient split sizes"],
            "rows": [],
        }
    suite = train_quantile_models(
        splits.train_df[splits.feature_cols].values,
        splits.train_df[splits.target_col].values,
        quantile_levels=FAST_LEVELS,
        xgb_params={**config["models"]["xgboost"], "n_estimators": 80, "subsample": 0.5},
        verbose=False,
    )
    cal_preds = suite.predict_sorted(splits.cal_b_df[splits.feature_cols].values)
    cal_a_preds = suite.predict_sorted(splits.cal_a_df[splits.feature_cols].values)
    tau_arr = np.full(len(splits.cal_b_df), tau)
    fam2 = train_family2_cal_a(cal_a_preds, splits.cal_a_df["Y"].to_numpy(), np.full(len(splits.cal_a_df), tau))
    p_hat = predict_family2(fam2, cal_preds, tau_arr)
    h = config["data"]["horizon_h"]
    ltt = run_ltt_certification(
        {q: cal_preds[q] for q in GATE_QS}, tau_arr, splits.cal_b_df["Y"].to_numpy(),
        splits.cal_b_df["window"].to_numpy() // h, cm, GATE_QS,
        alpha_fmr=0.10, delta_ltt=config["calibration"]["delta_ltt"],
        extent_ids=splits.cal_b_df["extent"].to_numpy(), verbose=False, p_hat=p_hat,
    )
    test_preds = suite.predict_sorted(splits.test_df[splits.feature_cols].values)
    test = attach_preds(splits.test_df, test_preds)
    test.attrs["extent_size_mib"] = config["cost_model"]["extent_size_mib"]
    capacity = max(1, int(n_ext * config["budget"]["ssd_fraction"]))
    budget = int(config["budget"]["max_migrations_per_epoch"])
    cell = ltt.best_cell
    if cell is not None and cell.family == "quantile" and cell.q == cell.q:
        qcol = f"q{int(round(cell.q * 100)):02d}"
        if qcol in test.columns:
            test["q_lo"] = test[qcol]
    oracle = run_baseline("hindsight_oracle", test, capacity, budget, cm, extent_size_mib=config["cost_model"]["extent_size_mib"])
    oracle_log = set(oracle.decision_log)
    rows = []
    proposed = run_baseline(
        "proposed", test, capacity, budget, cm,
        delta=0.0 if cell is None else float(cell.delta),
        extent_size_mib=config["cost_model"]["extent_size_mib"],
    )
    prow = metrics_from_replay(proposed, oracle_log, alpha=0.10)
    prow.update({"policy": "risk_controlled", "alpha": 0.10, "family": None if cell is None else cell.family})
    rows.append(prow)
    for policy in ("point_forecast", "static_threshold", "hdd_only", "all_ssd", "hindsight_oracle"):
        r = run_baseline(policy, test, capacity, budget, cm, threshold=5.0, extent_size_mib=config["cost_model"]["extent_size_mib"])
        row = metrics_from_replay(r, oracle_log)
        row.update({"policy": "oracle" if policy == "hindsight_oracle" else policy})
        rows.append(row)
    return {
        "volume": name,
        "header": header,
        "units": {
            "timestamp": "seconds (converted from Windows filetime in loader)",
            "extent": f"Offset // {config['data']['extent_size_bytes']} bytes",
            "size": "bytes",
        },
        "n_windows": n_windows,
        "n_extents": n_ext,
        "feature_rows": int(len(features)),
        "split_rows": splits.metadata["rows"],
        "tau_median": tau,
        "prevalence": prevalence,
        "notes": notes,
        "ltt": {
            "family": ltt.family_chosen,
            "n_units": ltt.n_cal_units,
            "delta_ltt": ltt.delta_ltt,
            "chosen": None if cell is None else {
                "q": cell.q, "delta": cell.delta, "p_star": cell.p_star,
                "certified": cell.certified, "activity": cell.migration_activity,
            },
        },
        "rows": rows,
    }


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    cache = ROOT / "results" / "msrc_16m_parquet"
    out_dir = ROOT / "results" / "jobs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for vol in ("hm_0", "prn_0"):
        path = out_dir / f"msrc_{vol}.json"
        if path.exists():
            print("skip", path.name, flush=True)
            continue
        parquet = cache / f"{vol}.parquet"
        if not parquet.exists():
            path.write_text(json.dumps({
                "volume": vol, "status": "missing",
                "notes": [f"no parquet at {parquet}; raw files also absent under data/msrc_raw"],
            }, indent=2))
            print("missing", vol, flush=True)
            continue
        payload = eval_volume(vol, parquet, config)
        path.write_text(json.dumps(payload, indent=2, default=str))
        print(path.name, "rows", payload.get("header", {}).get("rows"), "prevalence", payload.get("prevalence"), flush=True)


if __name__ == "__main__":
    main()
