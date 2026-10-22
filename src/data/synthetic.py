"""
Seeded synthetic storage workload generator.

Produces per-window, per-extent access counts with:
- Zipf head popularity with tens of accesses per window for hot extents
- Per-extent event generation: precursor-driven spikes, silent spikes, and decoy events
- Target 5% to 10% prevalence of (extent, epoch) pairs with Y >= nominal tau
- Steady workload: zero spikes, only low-rate decoy ramps
- Concept drift: hot-set rotation at configurable onset window
- Seeded via numpy.random.Generator from master seed (>= 2000 windows per seed)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class SpikeSpec:
    onset: int
    duration: int
    amplitude: float
    extents: np.ndarray
    spike_type: str       # 'precursor', 'silent', or 'plain'
    ramp_windows: int     # >0 for precursor
    extent: int = 0


@dataclass
class DecoySpec:
    onset: int
    ramp_windows: int
    fade_windows: int
    peak_amplitude: float
    extents: np.ndarray
    extent: int = 0


def _zipf_weights(n: int, alpha: float) -> np.ndarray:
    """Normalised Zipf weights for ranks 1..n."""
    ranks = np.arange(1, n + 1, dtype=np.float64)
    w = 1.0 / np.power(ranks, alpha)
    return w / w.sum()


def _generate_events_per_extent(
    rng: np.random.Generator,
    wtype: str,
    cfg: Dict[str, Any],
    n_windows: int,
    n_extents: int,
) -> Tuple[List[SpikeSpec], List[DecoySpec]]:
    """
    Generate spike and decoy events per extent according to workload type.
    - steady: 0 spikes, only low-rate decoys.
    - bursty: higher spike rate, precursor + silent mix + decoys.
    - read_heavy / changing_hot_set: moderate spike rate + decoys.
    - noisy_bursty: many decoys, fewer precursors, overdispersed counts.
    """
    sp_cfg = cfg.get("spikes_per_extent", {})
    dc_cfg = cfg.get("decoys_per_extent", {})

    amp_lo, amp_hi = sp_cfg.get("amplitude_range", [8.0, 25.0])
    dur_lo, dur_hi = sp_cfg.get("duration_range", [3, 10])
    ramp_w_sp = sp_cfg.get("precursor_ramp_windows", 4)
    precursor_frac = sp_cfg.get("precursor_fraction", 0.6)

    peak_lo, peak_hi = dc_cfg.get("peak_amplitude_range", [3.0, 7.0])
    ramp_w_dc = dc_cfg.get("ramp_windows", 4)
    fade_w_dc = dc_cfg.get("fade_windows", 4)

    if wtype == "steady":
        sp_rate = sp_cfg.get("steady_rate", 0.0)
        dc_rate = dc_cfg.get("steady_rate", 0.04)
    elif wtype == "bursty":
        sp_rate = sp_cfg.get("bursty_rate", 0.18)
        dc_rate = dc_cfg.get("bursty_rate", 0.08)
    elif wtype == "noisy_bursty":
        sp_rate = sp_cfg.get("noisy_bursty_rate", 0.18)
        dc_rate = dc_cfg.get("noisy_bursty_rate", 0.22)
    else:
        sp_rate = sp_cfg.get("default_rate", 0.15)
        dc_rate = dc_cfg.get("default_rate", 0.08)

    spikes: List[SpikeSpec] = []
    decoys: List[DecoySpec] = []

    for ext in range(n_extents):
        # Spikes on this extent
        if sp_rate > 0.0:
            n_sp = int(rng.poisson(sp_rate))
            for _ in range(n_sp):
                if wtype == "noisy_bursty" and "noisy_event_window_range" in cfg:
                    lo, hi = cfg["noisy_event_window_range"]
                    onset = int(rng.integers(lo, min(hi, n_windows - dur_hi - 5)))
                else:
                    onset = int(rng.integers(ramp_w_sp + 5, n_windows - dur_hi - 5))
                duration = int(rng.integers(dur_lo, dur_hi + 1))
                amplitude = float(rng.uniform(amp_lo, amp_hi))
                precursor_probability = (
                    sp_cfg.get("noisy_precursor_fraction", 0.3)
                    if wtype == "noisy_bursty" else precursor_frac
                )
                silent_probability = (
                    sp_cfg.get("noisy_silent_fraction", 0.4)
                    if wtype == "noisy_bursty" else 1.0 - precursor_probability
                )
                event_draw = rng.random()
                spike_type = (
                    "precursor" if event_draw < precursor_probability
                    else "silent" if event_draw < precursor_probability + silent_probability
                    else "plain"
                )
                spikes.append(SpikeSpec(
                    onset=onset,
                    duration=duration,
                    amplitude=amplitude,
                    extents=np.array([ext], dtype=np.int32),
                    spike_type=spike_type,
                    ramp_windows=ramp_w_sp if spike_type == "precursor" else 0,
                    extent=ext,
                ))

        # Decoys on this extent
        if dc_rate > 0.0:
            n_dc = int(rng.poisson(dc_rate))
            for _ in range(n_dc):
                if wtype == "noisy_bursty" and "noisy_event_window_range" in cfg:
                    lo, hi = cfg["noisy_event_window_range"]
                    onset = int(rng.integers(lo, min(hi, n_windows - fade_w_dc - 5)))
                else:
                    onset = int(rng.integers(ramp_w_dc + 5, n_windows - fade_w_dc - 5))
                peak_amp = float(rng.uniform(peak_lo, peak_hi))
                decoys.append(DecoySpec(
                    onset=onset,
                    ramp_windows=ramp_w_dc,
                    fade_windows=fade_w_dc,
                    peak_amplitude=peak_amp,
                    extents=np.array([ext], dtype=np.int32),
                    extent=ext,
                ))

    return spikes, decoys


def _apply_events_to_rates(
    rate_matrix: np.ndarray,
    spikes: List[SpikeSpec],
    decoys: List[DecoySpec],
) -> np.ndarray:
    """Apply spike and decoy rate boosts to extent rate matrix."""
    n_windows = rate_matrix.shape[0]

    # Spikes
    for sp in spikes:
        ext = sp.extent
        # Precursor ramp
        if sp.spike_type == "precursor" and sp.ramp_windows > 0:
            ramp_start = max(0, sp.onset - sp.ramp_windows)
            for w in range(ramp_start, sp.onset):
                frac = (w - ramp_start + 1) / sp.ramp_windows
                rate_matrix[w, ext] += sp.amplitude * frac * 0.4
        # Main spike duration
        for w in range(sp.onset, min(sp.onset + sp.duration, n_windows)):
            rate_matrix[w, ext] += sp.amplitude

    # Decoys: ramp up and fade out
    for d in decoys:
        ext = d.extent
        ramp_start = max(0, d.onset - d.ramp_windows)
        for w in range(ramp_start, min(d.onset, n_windows)):
            frac = (w - ramp_start + 1) / d.ramp_windows
            rate_matrix[w, ext] += d.peak_amplitude * frac
        for w in range(d.onset, min(d.onset + d.fade_windows, n_windows)):
            frac = 1.0 - (w - d.onset + 1) / d.fade_windows
            rate_matrix[w, ext] += d.peak_amplitude * max(frac, 0.0)

    return rate_matrix


def _apply_drift(
    rate_matrix: np.ndarray,
    rng: np.random.Generator,
    onset_window: int,
    hot_set_fraction: float,
) -> np.ndarray:
    """At onset_window, rotate hot_set_fraction of top extents to cold extents."""
    n_extents = rate_matrix.shape[1]
    pre_drift_means = rate_matrix[:onset_window].mean(axis=0)
    n_hot = max(1, int(n_extents * hot_set_fraction))
    hot_indices = np.argsort(pre_drift_means)[-n_hot:]
    cold_indices = np.argsort(pre_drift_means)[:n_hot]

    for w in range(onset_window, rate_matrix.shape[0]):
        hot_vals = rate_matrix[w, hot_indices].copy()
        cold_vals = rate_matrix[w, cold_indices].copy()
        rate_matrix[w, hot_indices] = cold_vals
        rate_matrix[w, cold_indices] = hot_vals
    return rate_matrix


def compute_epoch_y_stats(
    df: pd.DataFrame,
    n_extents: int = 500,
    n_windows: int = 2500,
    h: int = 5,
    tau_nominal: float = 5.0,
) -> Dict[str, Any]:
    """
    Compute distribution of Y (access count in next h windows) on non-overlapping
    decision epochs and prevalence of Y >= nominal tau across all (extent, epoch) pairs.
    """
    n_epochs = n_windows // h
    counts = np.zeros((n_windows, n_extents), dtype=np.int32)
    counts[df["window"].values, df["extent"].values] = df["count"].values

    y_epochs = []
    for ep in range(n_epochs - 1):
        w_start = (ep + 1) * h
        w_end = w_start + h
        y_ep = counts[w_start:w_end, :].sum(axis=0)
        y_epochs.append(y_ep)

    Y_all = np.concatenate(y_epochs)
    prev = float((Y_all >= tau_nominal).mean() * 100.0)
    med = float(np.median(Y_all))
    p90 = float(np.percentile(Y_all, 90))
    p99 = float(np.percentile(Y_all, 99))

    return {
        "tau_nominal": tau_nominal,
        "n_pairs": len(Y_all),
        "prevalence_pct": round(prev, 2),
        "median": round(med, 2),
        "p90": round(p90, 2),
        "p99": round(p99, 2),
    }


def generate_workload(
    config: Dict[str, Any],
    seed: int,
    workload_type: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate one synthetic workload realisation.
    """
    rng = np.random.default_rng(seed)
    cfg = config["data"]["synthetic"]
    n_extents: int = cfg["n_extents"]
    n_windows: int = cfg["n_windows"]
    window_sec: int = config["data"]["window_sec"]
    h: int = config["data"].get("horizon_h", 5)
    tau_nominal: float = config.get("cost_model", {}).get("nominal_tau", 5.0)
    wtype = workload_type or cfg.get("workload_type", "bursty")

    # Base rates: Zipf with tens of accesses per window for head extents
    weights = _zipf_weights(n_extents, cfg["zipf_alpha"])
    base_rates = weights * cfg["base_total_rate"]

    # Diurnal modulation
    t = np.arange(n_windows, dtype=np.float64)
    diurnal = 1.0 + cfg["diurnal_amplitude"] * np.sin(
        2.0 * np.pi * t / cfg["diurnal_period_windows"]
    )
    rate_matrix = np.outer(diurnal, base_rates)

    read_ratio: float = cfg["read_ratio"]
    if wtype == "read_heavy":
        read_ratio = 0.95

    # Events per extent (spikes and decoys)
    spikes, decoys = _generate_events_per_extent(rng, wtype, cfg, n_windows, n_extents)
    rate_matrix = _apply_events_to_rates(rate_matrix, spikes, decoys)

    # Drift (only for changing_hot_set)
    drift_info: Optional[Dict] = None
    if wtype == "changing_hot_set":
        drift_cfg = cfg["drift"]
        rate_matrix = _apply_drift(
            rate_matrix, rng,
            onset_window=drift_cfg["onset_window"],
            hot_set_fraction=drift_cfg["hot_set_fraction"],
        )
        drift_info = dict(drift_cfg)

    # Sample Poisson counts, or negative-binomial counts for the noisy preset.
    if wtype == "noisy_bursty":
        dispersion = float(cfg.get("noisy_dispersion", 2.0))
        mean_rates = np.maximum(rate_matrix, 0.0)
        p = dispersion / (dispersion + mean_rates)
        counts = rng.negative_binomial(dispersion, p)
    else:
        counts = rng.poisson(np.maximum(rate_matrix, 0.0))

    win_idx, ext_idx = np.nonzero(counts)
    vals = counts[win_idx, ext_idx]

    read_counts = rng.binomial(vals, read_ratio)
    write_counts = vals - read_counts

    mean_size = cfg["mean_request_size_bytes"]
    size_jitter = rng.uniform(0.7, 1.3, size=len(vals))

    df = pd.DataFrame({
        "window": win_idx.astype(np.int32),
        "extent": ext_idx.astype(np.int32),
        "count": vals.astype(np.int32),
        "read_count": read_counts.astype(np.int32),
        "write_count": write_counts.astype(np.int32),
        "mean_size_bytes": (mean_size * size_jitter).astype(np.float32),
        "timestamp_start_s": (win_idx * window_sec).astype(np.float64),
    })

    mean_lambda = float(counts.sum()) / n_windows
    y_stats = compute_epoch_y_stats(df, n_extents, n_windows, h=h, tau_nominal=tau_nominal)

    meta = {
        "seed": seed,
        "workload_type": wtype,
        "n_windows": n_windows,
        "n_extents": n_extents,
        "extent_size_bytes": int(config["data"].get("extent_size_bytes", 1024 * 1024)),
        "window_sec": window_sec,
        "mean_lambda": mean_lambda,
        "n_spikes": len(spikes),
        "n_precursor": sum(1 for s in spikes if s.spike_type == "precursor"),
        "n_silent": sum(1 for s in spikes if s.spike_type == "silent"),
        "n_plain": sum(1 for s in spikes if s.spike_type == "plain"),
        "n_decoys": len(decoys),
        "spikes": [
            {"onset": s.onset, "duration": s.duration,
             "amplitude": s.amplitude, "type": s.spike_type,
             "extent": s.extent, "n_affected": len(s.extents)}
            for s in spikes
        ],
        "drift": drift_info,
        "read_ratio": read_ratio,
        "y_stats": y_stats,
        "top5_base_rates": [float(r) for r in base_rates[:5].round(1)],
    }
    return df, meta


def generate_all_workloads(config: Dict[str, Any]) -> Dict[str, List[Tuple[pd.DataFrame, Dict]]]:
    """Generate workloads for all configured types and seeds."""
    n_seeds: int = config["n_seeds"]
    master_seed: int = config["seed"]
    wtypes: List[str] = config["data"]["synthetic"]["workload_types"]

    results: Dict[str, List[Tuple[pd.DataFrame, Dict]]] = {}
    for wtype in wtypes:
        results[wtype] = []
        for s_idx in range(n_seeds):
            seed = abs(master_seed + s_idx * 1000 + hash(wtype) % 1000) % (2**31)
            df, meta = generate_workload(config, seed=seed, workload_type=wtype)
            results[wtype].append((df, meta))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic workload generator")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Synthetic Storage Workload Generator (Per-Extent Events)")
    print("=" * 60)

    wtypes = config["data"]["synthetic"]["workload_types"]
    tau_nom = config.get("cost_model", {}).get("nominal_tau", 5.0)

    for wtype in wtypes:
        df, meta = generate_workload(config, seed=config["seed"], workload_type=wtype)
        ys = meta["y_stats"]
        print(f"\nWorkload: {wtype}")
        print(f"  Rows: {len(df):,}, Mean lambda: {meta['mean_lambda']:.1f} req/win")
        print(f"  Top 5 extent base rates (req/win): {meta['top5_base_rates']}")
        print(f"  Events: {meta['n_spikes']} spikes ({meta['n_precursor']} precursor, "
              f"{meta['n_silent']} silent), {meta['n_decoys']} decoys")
        print(f"  Prevalence Y >= {tau_nom}: {ys['prevalence_pct']:.2f}% (Target: 5-10%)")
        print(f"  Y Distribution: median={ys['median']}, p90={ys['p90']}, p99={ys['p99']}")


if __name__ == "__main__":
    main()
