"""
Learn-then-Test (LTT) gate certification.

Family 1: migrate iff q_hat_q(x) >= tau(x) * (1 + delta).
Family 2: migrate iff isotonic-calibrated P(Y >= tau | x) >= p*.

Nested testing walks from most conservative to least conservative inside a
family. Bonferroni spends delta_LTT over the score families actually tested
(at most 2), not over quantile levels. Fixed-sequence testing inside a family
needs no extra correction.

Calibration units are (epoch x extent-shard) with 8 shards from a stable hash
of extent id. Shards within an epoch are correlated, so the finite-sample
guarantee is approximate.

Hoeffding-Bentkus:
  p_H = exp(-2 * n * max(0, alpha_FMR - FMR_hat)^2)
  p_B = min(1, e * P(Bin(n, alpha_FMR) <= n * FMR_hat))
  p = min(p_H, p_B)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats


@dataclass
class GateCell:
    """A single cell in an LTT family grid."""
    q: float
    delta: float
    fdp_values: np.ndarray
    n_epochs_with_mig: int
    total_migrations: int
    wasted_migrations: int
    mean_fdp: float
    pooled_fmr: float
    p_value: float
    certified: bool
    net_benefit: float
    migration_activity: float
    family: str = "quantile"
    p_star: Optional[float] = None
    binomial_p_value: Optional[float] = None
    binomial_certified: bool = False
    n_cal_units: int = 0
    n_migrating_units: int = 0


@dataclass
class LTTResult:
    """Result of LTT gate certification."""
    grid: List[GateCell]
    certified_cells: List[GateCell]
    best_cell: Optional[GateCell]
    alpha_fmr: float
    delta_ltt: float
    n_cal_epochs: int
    n_cal_units: int
    K: int
    min_certifiable_alpha: float
    min_certifiable_alpha_bentkus: float
    certifiability_report: str
    n_cal_epochs_before_shards: int = 0
    n_families_tested: int = 1
    p_threshold: float = 0.10
    family_chosen: Optional[str] = None
    family_reports: Dict[str, Any] = field(default_factory=dict)
    silent_gate_warning: Optional[str] = None
    shard_correlation_note: str = (
        "Shard units within an epoch share the same time index and are "
        "correlated, so the LTT guarantee on (epoch x shard) units is approximate."
    )


def compute_certifiability(
    n_cal_units: int,
    K: int,
    delta_ltt: float,
) -> Tuple[float, float]:
    """Minimum certifiable alpha_FMR for Hoeffding and Bentkus bounds."""
    log_term = np.log(max(K, 1) / delta_ltt)
    alpha_hoeffding = np.sqrt(log_term / (2 * max(n_cal_units, 1)))
    alpha_bentkus = np.sqrt(np.log(np.e * max(K, 1) / delta_ltt) / (2 * max(n_cal_units, 1)))
    return float(alpha_hoeffding), float(alpha_bentkus)


def _hoeffding_bentkus_pvalue(
    fdp_hat: float,
    alpha_fmr: float,
    n: int,
) -> float:
    """Hoeffding-Bentkus p-value for H0: FMR > alpha_FMR vs small observed FMR."""
    if n <= 0:
        return 1.0
    gap = max(0.0, alpha_fmr - fdp_hat)
    p_h = np.exp(-2.0 * n * gap ** 2)
    k_obs = int(np.floor(n * fdp_hat))
    p_binom = sp_stats.binom.cdf(k_obs, n, alpha_fmr)
    p_b = min(1.0, np.e * p_binom)
    return float(min(p_h, p_b))


def clopper_pearson_pvalue(
    wasted: int,
    total: int,
    alpha_fmr: float,
) -> float:
    """
    Exact binomial (Clopper-Pearson) one-sided p-value on pooled counts.
    Tests whether observed wasted/total is compatible with FMR <= alpha.
    Labelled approximate because shard units within an epoch are correlated.
    """
    if total <= 0:
        return 1.0
    # P(Bin(total, alpha) <= wasted) ; small values support FMR <= alpha
    return float(sp_stats.binom.cdf(int(wasted), int(total), alpha_fmr))


def extent_shard_ids(extent_ids: np.ndarray, n_shards: int = 8) -> np.ndarray:
    """Stable hash of extent id into n_shards bins (not Python's salted hash())."""
    ids = np.asarray(extent_ids, dtype=np.int64)
    mixed = np.uint64(ids.astype(np.uint64)) * np.uint64(2654435761)
    return (mixed % np.uint64(max(1, n_shards))).astype(np.int64)


def make_unit_ids(
    epoch_ids: np.ndarray,
    extent_ids: Optional[np.ndarray],
    n_extent_shards: int = 8,
    seed_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, int]:
    """
    Build (seed x epoch x shard) unit ids.
    Returns unit_ids, n_units, n_epochs_before_shards.
    """
    epoch_ids = np.asarray(epoch_ids, dtype=np.int64)
    n_epochs = int(len(np.unique(epoch_ids)))
    if seed_ids is None:
        seed_ids = np.zeros(len(epoch_ids), dtype=np.int64)
    else:
        seed_ids = np.asarray(seed_ids, dtype=np.int64)
    if extent_ids is not None and n_extent_shards > 1:
        shards = extent_shard_ids(extent_ids, n_extent_shards)
        # Unique packing: seed in high bits, then epoch, then shard.
        unit_ids = seed_ids * np.int64(10_000_000) + epoch_ids * np.int64(n_extent_shards) + shards
    else:
        unit_ids = seed_ids * np.int64(10_000_000) + epoch_ids
    n_units = int(len(np.unique(unit_ids)))
    return unit_ids, n_units, n_epochs


def _realized_net_benefit(cost_model: Any, y: np.ndarray) -> np.ndarray:
    """Vectorized realized net benefit with V_evict=0, matching CostModel."""
    y = np.asarray(y, dtype=float)
    if hasattr(cost_model, "base_c_mig") and hasattr(cost_model, "delta_l"):
        fixed = (cost_model.base_c_mig * cost_model.rescale_factor) + cost_model.c_wb
        return y * cost_model.delta_l - fixed
    return np.array([cost_model.compute_realized_net_benefit(float(v)) for v in y], dtype=float)


def _compute_unit_fdps(
    migrate_mask: np.ndarray,
    Y_actual: np.ndarray,
    tau: np.ndarray,
    unit_ids: np.ndarray,
    cost_model: Any,
) -> Tuple[np.ndarray, int, int, int, float]:
    """Per-unit FDP for a boolean migrate mask."""
    unit_ids = np.asarray(unit_ids)
    migrate_mask = np.asarray(migrate_mask, dtype=bool)
    nb_all = _realized_net_benefit(cost_model, Y_actual)
    wasted_flag = migrate_mask & (nb_all <= 0.0)
    _, inv = np.unique(unit_ids, return_inverse=True)
    n_units = int(inv.max()) + 1 if len(inv) else 0
    mig_count = np.bincount(inv, weights=migrate_mask.astype(float), minlength=n_units)
    wasted_count = np.bincount(inv, weights=wasted_flag.astype(float), minlength=n_units)
    nb_sum = np.bincount(inv, weights=np.where(migrate_mask, nb_all, 0.0), minlength=n_units)
    fdps = np.zeros(n_units, dtype=float)
    active = mig_count > 0
    fdps[active] = wasted_count[active] / mig_count[active]
    return (
        fdps,
        int(active.sum()),
        int(mig_count.sum()),
        int(wasted_count.sum()),
        float(nb_sum.sum()),
    )


def _compute_epoch_fdps(
    Y_lo: np.ndarray,
    tau: np.ndarray,
    delta: float,
    Y_actual: np.ndarray,
    epoch_ids: np.ndarray,
    cost_model: Any,
) -> Tuple[np.ndarray, int, int, int, float]:
    """Backward-compatible quantile gate FDP helper."""
    threshold = tau * (1.0 + delta)
    migrate_mask = Y_lo >= threshold
    return _compute_unit_fdps(migrate_mask, Y_actual, tau, epoch_ids, cost_model)


def _cell_from_mask(
    migrate_mask: np.ndarray,
    Y_actual: np.ndarray,
    tau: np.ndarray,
    unit_ids: np.ndarray,
    cost_model: Any,
    alpha_fmr: float,
    p_threshold: float,
    sequence_broken: bool,
    q: float,
    delta: float,
    family: str,
    p_star: Optional[float],
) -> GateCell:
    fdps, n_with_mig, total_mig, wasted_mig, net_benefit = _compute_unit_fdps(
        migrate_mask, Y_actual, tau, unit_ids, cost_model
    )
    n_units = int(len(np.unique(unit_ids)))
    mean_fdp = float(np.mean(fdps)) if len(fdps) else 0.0
    pooled_fmr = wasted_mig / max(1, total_mig)
    activity = n_with_mig / max(1, n_units)
    p_val = _hoeffding_bentkus_pvalue(mean_fdp, alpha_fmr, n_units)
    binom_p = clopper_pearson_pvalue(wasted_mig, total_mig, alpha_fmr)
    is_certified = (
        p_val <= p_threshold
        and mean_fdp <= alpha_fmr
        and (not sequence_broken)
    )
    binom_certified = (not sequence_broken) and (total_mig > 0) and (binom_p <= p_threshold) and (pooled_fmr <= alpha_fmr)
    return GateCell(
        q=q,
        delta=round(float(delta), 4),
        fdp_values=fdps,
        n_epochs_with_mig=n_with_mig,
        total_migrations=total_mig,
        wasted_migrations=wasted_mig,
        mean_fdp=round(mean_fdp, 4),
        pooled_fmr=round(pooled_fmr, 4),
        p_value=round(p_val, 6),
        certified=bool(is_certified),
        net_benefit=round(net_benefit, 2),
        migration_activity=round(activity, 4),
        family=family,
        p_star=p_star,
        binomial_p_value=round(binom_p, 6),
        binomial_certified=bool(binom_certified),
        n_cal_units=n_units,
        n_migrating_units=n_with_mig,
    )


def _nested_select(cells: List[GateCell]) -> Optional[GateCell]:
    """Least conservative certified cell, then highest calibration net benefit."""
    certified = [c for c in cells if c.certified]
    if not certified:
        return None
    # Nested walk already ordered most->least conservative; last certified is least conservative.
    by_level: Dict[Tuple[str, float], GateCell] = {}
    for c in certified:
        key = (c.family, c.q if c.family != "probability" else -1.0)
        by_level[key] = c  # overwrite with later (less conservative) certified cell
    least = list(by_level.values())
    return max(least, key=lambda c: c.net_benefit)


def certify_quantile_family(
    quantile_preds: Dict[float, np.ndarray],
    tau_values: np.ndarray,
    Y_actual: np.ndarray,
    unit_ids: np.ndarray,
    cost_model: Any,
    gate_quantile_levels: Sequence[float],
    alpha_fmr: float,
    p_threshold: float,
    delta_grid: np.ndarray,
) -> List[GateCell]:
    """
    Family 1: for each q, walk delta from most conservative (large) to least.
    No extra Bonferroni across quantile levels; they are a nested sequence of
    scores inside one family.
    """
    all_cells: List[GateCell] = []
    for q in sorted(gate_quantile_levels):
        if q not in quantile_preds:
            continue
        Y_lo = quantile_preds[q]
        sequence_broken = False
        for delta in delta_grid:
            mask = Y_lo >= (tau_values * (1.0 + delta))
            cell = _cell_from_mask(
                mask, Y_actual, tau_values, unit_ids, cost_model,
                alpha_fmr, p_threshold, sequence_broken,
                q=float(q), delta=float(delta), family="quantile", p_star=None,
            )
            if not cell.certified:
                sequence_broken = True
            all_cells.append(cell)
    return all_cells


def certify_probability_family(
    p_hat: np.ndarray,
    tau_values: np.ndarray,
    Y_actual: np.ndarray,
    unit_ids: np.ndarray,
    cost_model: Any,
    alpha_fmr: float,
    p_threshold: float,
    p_star_grid: Sequence[float],
) -> List[GateCell]:
    """Family 2: walk p* from most conservative (high) to least (low)."""
    cells: List[GateCell] = []
    sequence_broken = False
    for p_star in sorted(p_star_grid, reverse=True):
        mask = p_hat >= float(p_star)
        cell = _cell_from_mask(
            mask, Y_actual, tau_values, unit_ids, cost_model,
            alpha_fmr, p_threshold, sequence_broken,
            q=float("nan"), delta=0.0, family="probability", p_star=float(p_star),
        )
        if not cell.certified:
            sequence_broken = True
        cells.append(cell)
    return cells


def certify_hybrid_family(
    quantile_preds: Dict[float, np.ndarray],
    p_hat: np.ndarray,
    tau_values: np.ndarray,
    Y_actual: np.ndarray,
    unit_ids: np.ndarray,
    cost_model: Any,
    q_level: float,
    alpha_fmr: float,
    p_threshold: float,
    t_grid: Sequence[float],
) -> List[GateCell]:
    """
    Hybrid: s(x) = max(q_hat_q(x) / tau(x), p_hat(x)); migrate if s >= t.
    Nested in t from high (conservative) to low.
    """
    q_level = float(q_level)
    Y_lo = quantile_preds[q_level]
    tau_safe = np.maximum(tau_values, 1e-6)
    score_q = Y_lo / tau_safe
    score = np.maximum(score_q, p_hat)
    cells: List[GateCell] = []
    sequence_broken = False
    for t in sorted(t_grid, reverse=True):
        mask = score >= float(t)
        cell = _cell_from_mask(
            mask, Y_actual, tau_values, unit_ids, cost_model,
            alpha_fmr, p_threshold, sequence_broken,
            q=q_level, delta=float(t) - 1.0, family="hybrid", p_star=float(t),
        )
        if not cell.certified:
            sequence_broken = True
        cells.append(cell)
    return cells


def run_ltt_certification(
    quantile_preds: Dict[float, np.ndarray],
    tau_values: np.ndarray,
    Y_actual: np.ndarray,
    epoch_ids: np.ndarray,
    cost_model: Any,
    gate_quantile_levels: List[float],
    alpha_fmr: float = 0.10,
    delta_ltt: float = 0.10,
    delta_range: Tuple[float, float] = (-0.8, 3.0),
    delta_steps: int = 39,
    n_extent_shards: int = 8,
    extent_ids: Optional[np.ndarray] = None,
    verbose: bool = True,
    p_hat: Optional[np.ndarray] = None,
    p_star_grid: Optional[Sequence[float]] = None,
    seed_ids: Optional[np.ndarray] = None,
    include_hybrid: bool = False,
    n_cal_epochs_before_shards: Optional[int] = None,
) -> LTTResult:
    """
    Run LTT certification on Cal-B (optionally pooled) data.

    Bonferroni: delta_LTT is split over families actually tested (1 or 2,
    plus hybrid only when requested as a fallback family).
    """
    unit_ids, n_cal_units, n_cal_epochs = make_unit_ids(
        epoch_ids, extent_ids, n_extent_shards, seed_ids=seed_ids
    )
    n_before = n_cal_epochs if n_cal_epochs_before_shards is None else int(n_cal_epochs_before_shards)

    families: List[str] = ["quantile"]
    if p_hat is not None:
        families.append("probability")
    if include_hybrid and p_hat is not None:
        families.append("hybrid")
    n_families = len(families)
    p_threshold = delta_ltt / n_families

    alpha_min_hoeff, alpha_min_bentkus = compute_certifiability(
        n_cal_units, n_families, delta_ltt
    )
    cert_report = (
        f"n_cal_epochs = {n_cal_epochs}, "
        f"n_cal_epochs_before_shards = {n_before}, "
        f"n_cal_units (epoch x shard) = {n_cal_units}, "
        f"n_families = {n_families}, "
        f"delta_LTT = {delta_ltt}, "
        f"p_threshold = {p_threshold:.4f}, "
        f"min certifiable alpha_FMR (Hoeffding) = {alpha_min_hoeff:.4f}, "
        f"min certifiable alpha_FMR (Bentkus) = {alpha_min_bentkus:.4f}. "
        "Shard units within an epoch are correlated; guarantee is approximate."
    )
    if verbose:
        print(f"\n  === LTT Certifiability Check ===")
        print(f"  {cert_report}")

    lo, hi = delta_range
    n_steps = int(round((hi - lo) / 0.1)) + 1 if delta_steps is None else int(delta_steps)
    # Most conservative (largest delta) to least conservative (smallest delta).
    delta_grid = np.round(np.arange(hi, lo - 1e-9, -0.1), 1)
    if len(delta_grid) == 0:
        delta_grid = np.linspace(hi, lo, max(2, n_steps))

    all_cells: List[GateCell] = []
    family_reports: Dict[str, Any] = {}

    q_cells = certify_quantile_family(
        quantile_preds, tau_values, Y_actual, unit_ids, cost_model,
        gate_quantile_levels, alpha_fmr, p_threshold, delta_grid,
    )
    all_cells.extend(q_cells)
    q_best = _nested_select(q_cells)
    family_reports["quantile"] = {
        "n_cells": len(q_cells),
        "n_certified": sum(1 for c in q_cells if c.certified),
        "best": None if q_best is None else {
            "q": q_best.q, "delta": q_best.delta, "net_benefit": q_best.net_benefit,
            "activity": q_best.migration_activity, "p_value": q_best.p_value,
            "migrating_units": q_best.n_migrating_units,
        },
    }

    p_best = None
    if p_hat is not None:
        if p_star_grid is None:
            p_star_grid = [round(x, 2) for x in np.arange(0.95, 0.04, -0.05)]
        p_cells = certify_probability_family(
            p_hat, tau_values, Y_actual, unit_ids, cost_model,
            alpha_fmr, p_threshold, p_star_grid,
        )
        all_cells.extend(p_cells)
        p_best = _nested_select(p_cells)
        family_reports["probability"] = {
            "n_cells": len(p_cells),
            "n_certified": sum(1 for c in p_cells if c.certified),
            "best": None if p_best is None else {
                "p_star": p_best.p_star, "net_benefit": p_best.net_benefit,
                "activity": p_best.migration_activity, "p_value": p_best.p_value,
                "migrating_units": p_best.n_migrating_units,
            },
        }

    h_best = None
    if include_hybrid and p_hat is not None:
        q_h = 0.10 if 0.10 in quantile_preds else sorted(quantile_preds)[0]
        t_grid = [round(x, 2) for x in np.arange(2.0, 0.09, -0.1)]
        h_cells = certify_hybrid_family(
            quantile_preds, p_hat, tau_values, Y_actual, unit_ids, cost_model,
            q_h, alpha_fmr, p_threshold, t_grid,
        )
        all_cells.extend(h_cells)
        h_best = _nested_select(h_cells)
        family_reports["hybrid"] = {
            "n_cells": len(h_cells),
            "n_certified": sum(1 for c in h_cells if c.certified),
            "best": None if h_best is None else {
                "q": h_best.q, "t": h_best.p_star, "net_benefit": h_best.net_benefit,
                "activity": h_best.migration_activity, "p_value": h_best.p_value,
            },
        }

    certified_cells = [c for c in all_cells if c.certified]
    candidates = [c for c in [q_best, p_best] if c is not None]
    if include_hybrid and h_best is not None:
        candidates.append(h_best)
    best_cell = max(candidates, key=lambda c: c.net_benefit) if candidates else None
    if best_cell is None and all_cells:
        best_cell = min(all_cells, key=lambda c: c.p_value)

    family_chosen = None if best_cell is None else best_cell.family
    warning = None
    if best_cell is None or (not best_cell.certified) or best_cell.migration_activity < 0.05:
        warning = (
            "SILENT GATE WARNING: certified cell has activity "
            f"{None if best_cell is None else best_cell.migration_activity} "
            "(< 0.05) or is uncertified. Policy still runs."
        )
        if verbose:
            print(f"  [!] {warning}")

    if verbose and best_cell is not None:
        print(
            f"  Family chosen={family_chosen} q={best_cell.q} delta={best_cell.delta} "
            f"p*={best_cell.p_star} certified={best_cell.certified} "
            f"activity={best_cell.migration_activity} nb={best_cell.net_benefit}"
        )

    return LTTResult(
        grid=all_cells,
        certified_cells=certified_cells,
        best_cell=best_cell,
        alpha_fmr=alpha_fmr,
        delta_ltt=delta_ltt,
        n_cal_epochs=n_cal_epochs,
        n_cal_units=n_cal_units,
        K=n_families,
        min_certifiable_alpha=alpha_min_hoeff,
        min_certifiable_alpha_bentkus=alpha_min_bentkus,
        certifiability_report=cert_report,
        n_cal_epochs_before_shards=n_before,
        n_families_tested=n_families,
        p_threshold=p_threshold,
        family_chosen=family_chosen,
        family_reports=family_reports,
        silent_gate_warning=warning,
    )


def format_ltt_summary(result: LTTResult) -> str:
    """Format a human-readable LTT summary."""
    lines = [
        "=== LTT Gate Certification Summary ===",
        result.certifiability_report,
        f"alpha_FMR = {result.alpha_fmr}, delta_LTT = {result.delta_ltt}",
        f"Certified cells: {len(result.certified_cells)} / {len(result.grid)}",
        f"Family chosen: {result.family_chosen}",
    ]
    if result.best_cell:
        bc = result.best_cell
        status = "CERTIFIED" if bc.certified else "NOT CERTIFIED (best uncertified)"
        lines.extend([
            f"\nBest cell: family={bc.family} q={bc.q} delta={bc.delta} p*={bc.p_star} [{status}]",
            f"  Mean FDP: {bc.mean_fdp:.4f}",
            f"  Pooled FMR: {bc.pooled_fmr:.4f}",
            f"  Net benefit: {bc.net_benefit:.1f}",
            f"  Migration activity: {bc.migration_activity:.3f}",
            f"  Migrating units: {bc.n_migrating_units} / {bc.n_cal_units}",
            f"  p-value (HB): {bc.p_value:.6f}",
            f"  p-value (binomial CP, approximate): {bc.binomial_p_value}",
        ])
    if result.silent_gate_warning:
        lines.append(result.silent_gate_warning)
    return "\n".join(lines)
