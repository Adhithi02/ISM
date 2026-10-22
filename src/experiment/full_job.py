"""One independent (seed, scenario) fast-replay job with pooled Cal-B certification."""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.calibration.defer import DeferConfig, DeferTracker
from src.calibration.ltt_gate import run_ltt_certification
from src.calibration.online_tracker import OnlineTracker, TrackerConfig
from src.data.preprocess import build_dense_feature_table, create_purged_time_splits
from src.data.synthetic import generate_workload
from src.models.prob_model import predict_family2, train_family2_cal_a
from src.models.quantile_xgb import train_quantile_models
from src.simulator.baselines import run_baseline
from src.simulator.cost_model import CostModel, CostModelConfig
from src.simulator.eviction import LRUEviction, ValueEviction
from src.simulator.fast_replay import ReplayMetrics, run_fast_replay

FAST_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.90]
GATE_QS = [0.05, 0.10, 0.20, 0.30, 0.50]
CAL_ONLY_OFFSET = 10_000


def make_cost_model(config: Dict[str, Any], cost_multiplier: float = 1.0) -> CostModel:
    cm = CostModel(CostModelConfig(
        l_hdd_ms=config["simulator"]["hdd"]["mean_service_ms"],
        l_ssd_ms=config["simulator"]["ssd"]["mean_service_ms"],
        dirty_fraction=config["cost_model"]["dirty_fraction"],
        extent_size_mib=config["cost_model"]["extent_size_mib"],
        kappa_interference=config["cost_model"]["kappa_interference"],
        measured_utilization=config["cost_model"]["measured_utilization"],
        ssd_wear_cost_per_mib=config["cost_model"]["ssd_wear_cost_per_mib"],
        c_mig_scale=cost_multiplier,
    ))
    return cm


def workload_type_for(scenario: str) -> str:
    return "noisy_bursty" if scenario == "noisy_bursty" else "changing_hot_set"


def generate_splits(config: Dict[str, Any], seed: int, scenario: str):
    raw, meta = generate_workload(config, seed=seed, workload_type=workload_type_for(scenario))
    features = build_dense_feature_table(
        raw, meta["n_windows"], meta["n_extents"], config["data"]["horizon_h"],
        extent_size_bytes=int(config["data"]["extent_size_bytes"]),
    )
    splits = create_purged_time_splits(features, meta["n_windows"], config)
    return splits, meta, features, raw


def attach_preds(df: pd.DataFrame, preds: Dict[float, np.ndarray]) -> pd.DataFrame:
    out = df.copy()
    for q, arr in preds.items():
        out[f"q{int(round(q * 100)):02d}"] = arr
    out["q50"] = preds[0.50]
    out["q_lo"] = preds.get(0.10, preds[0.50])
    return out


def metrics_from_replay(
    result: ReplayMetrics,
    oracle_log: Optional[set] = None,
    alpha: Optional[float] = None,
) -> Dict[str, Any]:
    n_ep = max(1, result.total_epochs)
    fdp = np.asarray(result.epoch_fdp_series, dtype=float)
    mig = np.asarray(result.epoch_migrations, dtype=float)
    frac_above = None
    if alpha is not None:
        mask = mig > 0
        if mask.any():
            frac_above = float(np.mean(fdp[mask] > alpha))
        else:
            frac_above = 0.0
    recall = None
    if oracle_log is not None:
        rec = set(result.decision_log).intersection(oracle_log)
        recall = (len(rec) / len(oracle_log)) if oracle_log else 0.0
    return {
        "migrations": result.total_migrations,
        "wasted": result.total_false_migrations,
        "pooled_fmr": result.pooled_fmr,
        "mean_fdp": result.mean_fdp,
        "activity": result.migration_activity,
        "net_benefit": result.total_net_benefit,
        "recall": recall,
        "frac_seed_blocks_fdp_gt_alpha": frac_above,
        "epoch_fdp": [float(x) for x in result.epoch_fdp_series],
        "epoch_migrations": [int(x) for x in result.epoch_migrations],
        "epoch_net_benefits": [float(x) for x in result.epoch_net_benefits],
        "n_epochs": result.total_epochs,
        "n_epochs_with_mig": result.active_epochs,
    }


def cell_payload(cell) -> Optional[Dict[str, Any]]:
    if cell is None:
        return None
    return {
        "family": cell.family,
        "q": None if cell.q != cell.q else cell.q,  # NaN -> None
        "delta": cell.delta,
        "p_star": cell.p_star,
        "p_value": cell.p_value,
        "binomial_p_value": cell.binomial_p_value,
        "binomial_certified": cell.binomial_certified,
        "certified": cell.certified,
        "mean_fdp": cell.mean_fdp,
        "pooled_fmr": cell.pooled_fmr,
        "net_benefit": cell.net_benefit,
        "activity": cell.migration_activity,
        "total_migrations": cell.total_migrations,
        "n_migrating_units": cell.n_migrating_units,
        "n_cal_units": cell.n_cal_units,
    }


def pool_cal_b(
    config: Dict[str, Any],
    scenario: str,
    model,
    current_splits,
    current_seed: int,
    train_seeds: List[int],
    n_cal_only: int,
    tau_value: float,
) -> Dict[str, np.ndarray]:
    """Pool Cal-B across training seeds + extra calibration-only seeds using this seed's model."""
    h = config["data"]["horizon_h"]
    chunks = []

    def add_cal(cal_df: pd.DataFrame, seed: int) -> None:
        if cal_df is None or len(cal_df) == 0:
            return
        preds = model.predict_sorted(cal_df[current_splits.feature_cols].values)
        chunks.append({
            "Y": cal_df["Y"].to_numpy(),
            "epoch": (cal_df["window"].to_numpy() // h).astype(np.int64),
            "extent": cal_df["extent"].to_numpy().astype(np.int64),
            "seed": np.full(len(cal_df), seed, dtype=np.int64),
            "preds": preds,
        })

    add_cal(current_splits.cal_b_df, current_seed)
    for seed in train_seeds:
        if seed == current_seed:
            continue
        splits, _, _, _ = generate_splits(config, seed, scenario)
        add_cal(splits.cal_b_df, seed)
    cheap_cfg = copy.deepcopy(config)
    cheap_cfg["data"]["synthetic"]["n_windows"] = min(
        500, int(config["data"]["synthetic"]["n_windows"])
    )
    for i in range(n_cal_only):
        seed = CAL_ONLY_OFFSET + current_seed * 100 + i
        splits, _, _, _ = generate_splits(cheap_cfg, seed, scenario)
        add_cal(splits.cal_b_df, seed)

    Y = np.concatenate([c["Y"] for c in chunks])
    epoch = np.concatenate([c["epoch"] for c in chunks])
    extent = np.concatenate([c["extent"] for c in chunks])
    seed_ids = np.concatenate([c["seed"] for c in chunks])
    preds_pool: Dict[float, np.ndarray] = {}
    for q in FAST_LEVELS:
        preds_pool[q] = np.concatenate([c["preds"][q] for c in chunks])
    tau = np.full(len(Y), tau_value)
    n_epochs_before = int(len(np.unique(current_splits.cal_b_df["window"].to_numpy() // h)))
    return {
        "Y": Y, "epoch": epoch, "extent": extent, "seed_ids": seed_ids,
        "preds": preds_pool, "tau": tau, "n_epochs_before": n_epochs_before,
        "n_rows": int(len(Y)), "n_source_chunks": len(chunks),
    }


def apply_defer_mask(
    df: pd.DataFrame,
    gate_fn: Callable,
    p_hat: np.ndarray,
    p_defer: float,
    max_defer: int,
    tau_value: float,
    delta: float,
) -> np.ndarray:
    tracker = DeferTracker(DeferConfig(p_defer=p_defer, max_defer_epochs=max_defer))
    y_lo = df["q_lo"].to_numpy()
    df = df.copy()
    df["_p_hat"] = p_hat
    migrate = np.zeros(len(df), dtype=bool)
    for w, idx in df.groupby("window").groups.items():
        idx = np.asarray(list(idx))
        sl = df.loc[idx]
        passed = gate_fn(sl, tau_value)
        ylo = sl["q_lo"].to_numpy()
        thresh = np.full(len(sl), tau_value * (1.0 + delta))
        # Force Y_lo above threshold where the family gate passed.
        ylo_adj = np.where(np.asarray(passed), thresh + 1.0, ylo)
        decisions = tracker.decide(
            sl["extent"].to_numpy(), ylo_adj, thresh, sl["_p_hat"].to_numpy(), int(w)
        )
        migrate[idx] = np.array([d.action == "migrate" for d in decisions])
    return migrate


def run_family_policy(
    test: pd.DataFrame,
    cell,
    p_hat_test: np.ndarray,
    capacity: int,
    budget: int,
    cm: CostModel,
    extent_size: float,
    online: bool,
    defer: bool,
    alpha: float,
    config: Dict[str, Any],
    eviction: str = "value",
) -> ReplayMetrics:
    df = test.copy()
    ev = ValueEviction() if eviction == "value" else LRUEviction()
    tracker_cfg = TrackerConfig(
        eta=config["online_tracker"]["eta"],
        delta_min=config["online_tracker"]["delta_min"],
        delta_max=config["online_tracker"]["delta_max"],
        h=config["data"]["horizon_h"],
    )

    if cell is None:
        gate_fn = lambda c, tau: pd.Series(False, index=c.index)
        return run_fast_replay(df, capacity, budget, cm, ev, gate_fn=gate_fn, extent_size_mib=extent_size)

    if cell.family == "probability":
        p_star = float(cell.p_star if cell.p_star is not None else 0.5)
        df["_p_hat"] = p_hat_test
        base_fn = lambda c, tau, ps=p_star: c["_p_hat"] >= ps
        init = p_star
        def provider_factory(tr: OnlineTracker):
            return lambda: tr.current_delta
        # Online on p* uses the same additive tracker in p*-space via delta_min/max clip.
        tracker = OnlineTracker(tracker_cfg, initial_delta=init, alpha_fmr=alpha)
        tracker.cfg.delta_min = 0.05
        tracker.cfg.delta_max = 0.99
        if online:
            gate_fn = lambda c, tau, tr=tracker: c["_p_hat"] >= tr.current_delta
            after = lambda w, m, fdp, tr=tracker: tr.update(w, fdp)
            return run_fast_replay(
                df, capacity, budget, cm, ev, gate_fn=gate_fn,
                after_epoch=after, extent_size_mib=extent_size,
            )
        if defer:
            mask = apply_defer_mask(
                df, base_fn, p_hat_test,
                config["defer"]["p_defer"], config["defer"]["max_defer_epochs"],
                cm.compute_tau(None), 0.0,
            )
            df["_defer"] = mask
            gate_fn = lambda c, tau: c["_defer"]
            return run_fast_replay(df, capacity, budget, cm, ev, gate_fn=gate_fn, extent_size_mib=extent_size)
        return run_fast_replay(df, capacity, budget, cm, ev, gate_fn=base_fn, extent_size_mib=extent_size)

    # Quantile or hybrid: store the certified score in q_lo.
    q_use = 0.10 if cell.q != cell.q else float(cell.q)
    qcol = f"q{int(round(q_use * 100)):02d}"
    if cell.family == "hybrid":
        tau_v = cm.compute_tau(None)
        score_q = df[qcol].to_numpy() / max(tau_v, 1e-6)
        score = np.maximum(score_q, p_hat_test)
        t = float(cell.p_star if cell.p_star is not None else 1.0)
        df["_hyb"] = score
        base_fn = lambda c, tau, tt=t: c["_hyb"] >= tt
        if online:
            tr = OnlineTracker(tracker_cfg, initial_delta=t, alpha_fmr=alpha)
            tr.cfg.delta_min = 0.1
            tr.cfg.delta_max = 2.0
            gate_fn = lambda c, tau, tracker=tr: c["_hyb"] >= tracker.current_delta
            after = lambda w, m, fdp, tracker=tr: tracker.update(w, fdp)
            return run_fast_replay(df, capacity, budget, cm, ev, gate_fn=gate_fn, after_epoch=after, extent_size_mib=extent_size)
        return run_fast_replay(df, capacity, budget, cm, ev, gate_fn=base_fn, extent_size_mib=extent_size)

    df["q_lo"] = df[qcol] if qcol in df.columns else df["q_lo"]
    delta0 = float(cell.delta)
    if defer:
        base_fn = lambda c, tau, d=delta0: c["q_lo"] >= tau * (1.0 + d)
        mask = apply_defer_mask(
            df, base_fn, p_hat_test,
            config["defer"]["p_defer"], config["defer"]["max_defer_epochs"],
            cm.compute_tau(None), delta0,
        )
        df["_defer"] = mask
        gate_fn = lambda c, tau: c["_defer"]
        return run_fast_replay(df, capacity, budget, cm, ev, gate_fn=gate_fn, extent_size_mib=extent_size)
    if online:
        tr = OnlineTracker(tracker_cfg, initial_delta=delta0, alpha_fmr=alpha)
        after = lambda w, m, fdp, tracker=tr: tracker.update(w, fdp)
        return run_fast_replay(
            df, capacity, budget, cm, ev, delta=delta0,
            delta_provider=lambda tracker=tr: tracker.current_delta,
            after_epoch=after, extent_size_mib=extent_size,
        )
    return run_fast_replay(df, capacity, budget, cm, ev, delta=delta0, extent_size_mib=extent_size)


def run_one_job(
    config: Dict[str, Any],
    seed: int,
    scenario: str,
    n_eval_seeds: int = 5,
    n_cal_only: int = 20,
    include_hybrid: bool = False,
    reduced: bool = False,
    ofat: bool = True,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    master = int(cfg["seed"])
    train_seeds = list(range(master, master + n_eval_seeds))
    splits, meta, features, raw = generate_splits(cfg, seed, scenario)
    xgb = {**cfg["models"]["xgboost"], "n_estimators": 100, "subsample": 0.5, "n_jobs": 2}
    suite = train_quantile_models(
        splits.train_df[splits.feature_cols].values,
        splits.train_df[splits.target_col].values,
        quantile_levels=FAST_LEVELS,
        xgb_params=xgb,
        verbose=False,
    )
    cm = make_cost_model(cfg)
    cm.check_and_rescale_tau_distribution([0.0] * min(300, len(splits.train_df)))
    tau_value = cm.compute_tau(None)

    cal_a_preds = suite.predict_sorted(splits.cal_a_df[splits.feature_cols].values)
    tau_cal_a = np.full(len(splits.cal_a_df), tau_value)
    fam2 = train_family2_cal_a(cal_a_preds, splits.cal_a_df["Y"].to_numpy(), tau_cal_a, verbose=False)

    pooled = pool_cal_b(cfg, scenario, suite, splits, seed, train_seeds, n_cal_only, tau_value)
    p_hat_cal = predict_family2(fam2, pooled["preds"], pooled["tau"])

    test_preds = suite.predict_sorted(splits.test_df[splits.feature_cols].values)
    if scenario == "stress":
        # Stress = drift workload plus noisy forecasts; noise is applied to
        # predictions only and is not a workload retune.
        rng_n = np.random.default_rng(seed + 17)
        for q in list(test_preds.keys()):
            scale = rng_n.lognormal(mean=0.0, sigma=0.20, size=len(test_preds[q]))
            test_preds[q] = np.clip(test_preds[q] * scale, 0.0, None)
    test = attach_preds(splits.test_df, test_preds)
    test.attrs["extent_size_mib"] = cfg["cost_model"]["extent_size_mib"]
    p_hat_test = predict_family2(fam2, test_preds, np.full(len(test), tau_value))
    test["p_hat"] = p_hat_test

    capacity = max(1, int(meta["n_extents"] * cfg["budget"]["ssd_fraction"]))
    budget = int(cfg["budget"]["max_migrations_per_epoch"])
    extent_size = cfg["cost_model"]["extent_size_mib"]

    oracle = run_baseline("hindsight_oracle", test, capacity, budget, cm, extent_size_mib=extent_size)
    oracle_log = set(oracle.decision_log)

    cert = {}
    rows = []
    silent_flags = []
    for alpha in cfg["experiment"]["alpha_sweep"]:
        ltt = run_ltt_certification(
            {q: pooled["preds"][q] for q in GATE_QS},
            pooled["tau"], pooled["Y"], pooled["epoch"], cm, GATE_QS,
            alpha_fmr=alpha, delta_ltt=cfg["calibration"]["delta_ltt"],
            delta_range=(-0.8, 3.0), delta_steps=39,
            n_extent_shards=8, extent_ids=pooled["extent"],
            verbose=False, p_hat=p_hat_cal, seed_ids=pooled["seed_ids"],
            include_hybrid=include_hybrid,
            n_cal_epochs_before_shards=pooled["n_epochs_before"],
        )
        best = ltt.best_cell
        cert[str(alpha)] = {
            "delta_ltt": ltt.delta_ltt,
            "p_threshold": ltt.p_threshold,
            "n_families_tested": ltt.n_families_tested,
            "n_cal_units": ltt.n_cal_units,
            "n_cal_epochs_before_shards": ltt.n_cal_epochs_before_shards,
            "n_cal_epochs": ltt.n_cal_epochs,
            "family_chosen": ltt.family_chosen,
            "family_reports": ltt.family_reports,
            "certifiability_report": ltt.certifiability_report,
            "chosen": cell_payload(best),
            "hybrid": cell_payload(next((c for c in ltt.certified_cells if c.family == "hybrid"), None))
            if include_hybrid else None,
            "smallest_p_value": min((c.p_value for c in ltt.grid), default=None),
            "n_certified": len(ltt.certified_cells),
            "silent_warning": ltt.silent_gate_warning,
            "shard_note": ltt.shard_correlation_note,
        }
        for online in (False, True):
            for defer in (False, True):
                if online and defer:
                    continue
                replay = run_family_policy(
                    test, best, p_hat_test, capacity, budget, cm, extent_size,
                    online=online, defer=defer, alpha=alpha, config=cfg,
                )
                row = metrics_from_replay(replay, oracle_log, alpha=alpha)
                row.update({
                    "policy": "risk_controlled",
                    "alpha": alpha,
                    "threshold_mode": "online" if online else "static",
                    "defer": defer,
                    "family": None if best is None else best.family,
                    "q": None if best is None or best.q != best.q else best.q,
                    "delta": None if best is None else best.delta,
                    "p_star": None if best is None else best.p_star,
                    "certified": False if best is None else best.certified,
                })
                rows.append(row)
                if (not online) and (not defer):
                    if row["activity"] < 0.05 or (
                        row["recall"] is not None and row["recall"] < 0.25
                    ):
                        silent_flags.append({
                            "alpha": alpha, "activity": row["activity"], "recall": row["recall"],
                            "warning": "silent or low-recall certified gate; policy still ran",
                        })

    point = run_baseline("point_forecast", test, capacity, budget, cm, extent_size_mib=extent_size)
    point_row = metrics_from_replay(point, oracle_log)
    point_recall = point_row["recall"] or 0.0
    for row in rows:
        if row["policy"] == "risk_controlled" and row["threshold_mode"] == "static" and not row["defer"]:
            if row["recall"] is not None and point_recall > 0 and row["recall"] < 0.25 * point_recall:
                silent_flags.append({
                    "alpha": row["alpha"],
                    "recall": row["recall"],
                    "point_recall": point_recall,
                    "warning": "recall < 25% of point_forecast; policy still ran",
                })

    baselines = (
        "point_forecast", "safety_buffer", "static_threshold", "lru_promote",
        "hdd_only", "all_ssd", "hindsight_oracle",
        "proposed_lru_evict", "point_forecast_value_evict",
    )
    for policy in baselines:
        result = run_baseline(
            policy, test, capacity, budget, cm,
            threshold=5.0, margin=2.0, extent_size_mib=extent_size,
        )
        row = metrics_from_replay(result, oracle_log)
        row.update({"policy": policy if policy != "hindsight_oracle" else "oracle",
                    "alpha": None, "threshold_mode": "static", "defer": False})
        rows.append(row)

    ofat_rows = []
    if ofat:
        alpha = 0.10
        cell = None
        chosen = cert.get("0.1", cert.get("0.10", {})).get("chosen")
        # Rebuild cell-like namespace from payload via a simple namespace.
        from types import SimpleNamespace
        if not chosen:
            cell = None
        else:
            payload = dict(chosen)
            if payload.get("q") is None:
                payload["q"] = float("nan")
            cell = SimpleNamespace(**payload)
        for cal_on in (True, False):
            if not cal_on:
                # Calibration off: q50 >= tau (uncertified point rule) labelled calibration=off.
                replay = run_baseline("point_forecast", test, capacity, budget, cm, extent_size_mib=extent_size)
                row = metrics_from_replay(replay, oracle_log, alpha=alpha)
                row.update({"factor": "calibration", "value": "off", "policy": "risk_controlled", "alpha": alpha})
                ofat_rows.append(row)
            else:
                replay = run_family_policy(test, cell, p_hat_test, capacity, budget, cm, extent_size, False, False, alpha, cfg)
                row = metrics_from_replay(replay, oracle_log, alpha=alpha)
                row.update({"factor": "calibration", "value": "on", "policy": "risk_controlled", "alpha": alpha})
                ofat_rows.append(row)
        for mode, online in (("static", False), ("online", True)):
            replay = run_family_policy(test, cell, p_hat_test, capacity, budget, cm, extent_size, online, False, alpha, cfg)
            row = metrics_from_replay(replay, oracle_log, alpha=alpha)
            row.update({"factor": "threshold_mode", "value": mode, "policy": "risk_controlled", "alpha": alpha})
            ofat_rows.append(row)
        for defer in (True, False):
            replay = run_family_policy(test, cell, p_hat_test, capacity, budget, cm, extent_size, False, defer, alpha, cfg)
            row = metrics_from_replay(replay, oracle_log, alpha=alpha)
            row.update({"factor": "defer", "value": "on" if defer else "off", "policy": "risk_controlled", "alpha": alpha})
            ofat_rows.append(row)
        for frac in cfg["experiment"]["ssd_sweep"]:
            cap = max(1, int(meta["n_extents"] * frac))
            replay = run_family_policy(test, cell, p_hat_test, cap, budget, cm, extent_size, False, False, alpha, cfg)
            row = metrics_from_replay(replay, oracle_log, alpha=alpha)
            row.update({"factor": "ssd_fraction", "value": frac, "policy": "risk_controlled", "alpha": alpha})
            ofat_rows.append(row)
        for mult in cfg["experiment"]["cost_sweep"]:
            cm_m = make_cost_model(cfg, cost_multiplier=mult)
            cm_m.rescale_factor = cm.rescale_factor
            replay = run_family_policy(test, cell, p_hat_test, capacity, budget, cm_m, extent_size, False, False, alpha, cfg)
            row = metrics_from_replay(replay, oracle_log, alpha=alpha)
            row.update({"factor": "cost_multiplier", "value": mult, "policy": "risk_controlled", "alpha": alpha})
            ofat_rows.append(row)

    prevalence = float(np.mean(test["Y"].to_numpy() >= tau_value))
    return {
        "seed": seed,
        "scenario": scenario,
        "reduced": reduced,
        "model_config": {"quantiles": FAST_LEVELS, "n_estimators": 100, "subsample": 0.5, "n_jobs": 2},
        "cost_breakdown": cm.cost_breakdown(),
        "tau": {"median": tau_value, "p10": tau_value, "p90": tau_value},
        "prevalence": prevalence,
        "oracle_useful": len(oracle_log),
        "pooled_cal": {
            "n_rows": pooled["n_rows"],
            "n_source_chunks": pooled["n_source_chunks"],
            "n_cal_only_seeds": n_cal_only,
            "n_epochs_before_shards": pooled["n_epochs_before"],
            "n_train_seeds": len(train_seeds),
        },
        "rows": rows,
        "ofat": ofat_rows,
        "certifiability": cert,
        "silent_flags": silent_flags,
        "include_hybrid": include_hybrid,
        "drift_onset_window": cfg["data"]["synthetic"]["drift"]["onset_window"],
        "n_windows": meta["n_windows"],
        "test_windows": [int(splits.test_df["window"].min()), int(splits.test_df["window"].max())],
    }
