"""
Vectorized, SimPy-free fast storage trace replay.

Tracks:
- SSD resident extent set and capacity
- Pluggable eviction (ValueEviction or LRUEviction)
- Migration budget (max M migrations per decision epoch)
- State-dependent tau(x)
- Per-epoch FDP (0 when nothing migrates), mean FDP, pooled FMR, and net benefit

Used for Cal-B calibration grid search, online tracker simulation, and OFAT ablations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple  # noqa: F401

import numpy as np
import pandas as pd

from src.simulator.cost_model import CostModel
from src.simulator.eviction import EvictionPolicy, ValueEviction, LRUEviction


@dataclass
class ReplayMetrics:
    total_epochs: int
    active_epochs: int
    migration_activity: float
    total_migrations: int
    total_false_migrations: int
    mean_fdp: float
    pooled_fmr: float
    total_net_benefit: float
    mean_net_benefit_per_epoch: float
    epoch_fdp_series: List[float]
    epoch_migrations: List[int]
    epoch_net_benefits: List[float]
    decision_log: List[Tuple[int, int]] = field(default_factory=list)


def run_fast_replay(
    decision_df: pd.DataFrame,
    ssd_capacity: int,
    max_migrations_per_epoch: int,
    cost_model: CostModel,
    eviction_policy: EvictionPolicy,
    gate_fn: Optional[Callable[[pd.DataFrame, float], pd.Series]] = None,
    delta: float = 0.0,
    q_lo_col: str = "q_lo",
    q50_col: str = "q50",
    y_col: str = "Y",
    extent_size_mib: float = 1.0,
    delta_provider: Optional[Callable[[], float]] = None,
    after_epoch: Optional[Callable[[int, int, float], None]] = None,
) -> ReplayMetrics:
    """
    Run fast vectorised replay across decision epochs.

    Parameters
    ----------
    decision_df : DataFrame
        Must contain columns: 'window', 'extent', y_col, and optionally q_lo_col, q50_col.
    ssd_capacity : int
        Maximum number of extents SSD can hold.
    max_migrations_per_epoch : int
        Maximum extents that can be migrated in one epoch.
    cost_model : CostModel
        Cost model for computing state-dependent tau and realized net benefit.
    eviction_policy : EvictionPolicy
        Eviction strategy (LRU or Value).
    gate_fn : Callable, optional
        Custom gate boolean mask: gate_fn(epoch_df, tau). If None, defaults to:
        (epoch_df[q_lo_col] >= tau * (1.0 + delta)).
    """
    if "extent_size_mib" in decision_df.attrs:
        extent_size_mib = float(decision_df.attrs["extent_size_mib"])
    ssd_residents: Set[int] = set()
    epoch_windows = sorted(decision_df["window"].unique())

    epoch_fdp_list: List[float] = []
    epoch_migs_list: List[int] = []
    epoch_net_benefit_list: List[float] = []
    decision_log: List[Tuple[int, int]] = []

    total_migrations = 0
    total_false_migrations = 0
    total_net_benefit = 0.0

    for w in epoch_windows:
        epoch_slice = decision_df[decision_df["window"] == w].copy()
        pred_dict = (
            dict(zip(epoch_slice["extent"], epoch_slice[q50_col]))
            if q50_col in epoch_slice.columns
            else {}
        )
        realized_dict = dict(zip(epoch_slice["extent"], epoch_slice[y_col]))

        # Update eviction policy with any hits on SSD
        for resident in list(ssd_residents):
            if resident in realized_dict and realized_dict[resident] > 0:
                eviction_policy.record_access(resident, timestamp=float(w))

        # Identify candidate extents currently on HDD
        candidates = epoch_slice[~epoch_slice["extent"].isin(ssd_residents)].copy()

        migrations_this_epoch = []
        evictions_this_epoch = []

        if not candidates.empty:
            # Evaluate candidates for migration
            # State-dependent tau: if SSD has room, evictee value is 0
            has_free_room = len(ssd_residents) < ssd_capacity
            if has_free_room:
                tau_base = cost_model.compute_tau(evictee_pred_q50=None)
            else:
                # Potential evictee prediction
                try:
                    evictee = eviction_policy.select_evictee(ssd_residents, pred_dict)
                    evictee_pred = pred_dict.get(evictee, 0.0)
                except ValueError:
                    evictee_pred = 0.0
                tau_base = cost_model.compute_tau(evictee_pred_q50=evictee_pred)

            live_delta = float(delta_provider()) if delta_provider is not None else float(delta)
            effective_threshold = tau_base * (1.0 + live_delta)

            if gate_fn is not None:
                passed_gate = gate_fn(candidates, tau_base)
            elif q_lo_col in candidates.columns:
                passed_gate = candidates[q_lo_col] >= effective_threshold
            elif q50_col in candidates.columns:
                passed_gate = candidates[q50_col] >= effective_threshold
            else:
                passed_gate = candidates[y_col] >= effective_threshold

            eligible = candidates[passed_gate]

            if not eligible.empty:
                # Rank by pessimistic surplus or predicted value
                rank_col = q_lo_col if q_lo_col in eligible.columns else (q50_col if q50_col in eligible.columns else y_col)
                sorted_eligible = eligible.sort_values(rank_col, ascending=False)
                selected = sorted_eligible.head(max_migrations_per_epoch)

                for _, row in selected.iterrows():
                    cand_ext = int(row["extent"])
                    cand_pred = float(row.get(q50_col, row[rank_col]))

                    # Check if eviction is needed
                    evictee_ext: Optional[int] = None
                    if len(ssd_residents) >= ssd_capacity:
                        evictee_ext = eviction_policy.select_evictee(ssd_residents, pred_dict)
                        ssd_residents.remove(evictee_ext)
                        eviction_policy.remove_extent(evictee_ext)
                        evictions_this_epoch.append(evictee_ext)

                    ssd_residents.add(cand_ext)
                    eviction_policy.add_extent(cand_ext, timestamp=float(w), predicted_value=cand_pred)
                    migrations_this_epoch.append((cand_ext, evictee_ext))
                    decision_log.append((int(w), cand_ext))

        # Evaluate performance of migrations in this epoch
        m_t = len(migrations_this_epoch)
        if m_t > 0:
            false_migs = 0
            epoch_net_ben = 0.0
            for cand_ext, evictee_ext in migrations_this_epoch:
                y_cand = realized_dict.get(cand_ext, 0.0)
                y_evict = realized_dict.get(evictee_ext, 0.0) if evictee_ext is not None else None
                net_ben = cost_model.compute_realized_net_benefit(y_cand, y_evict)
                epoch_net_ben += net_ben
                if net_ben <= 0.0:
                    false_migs += 1

            fdp_t = float(false_migs / m_t)
            total_migrations += m_t
            total_false_migrations += false_migs
            total_net_benefit += epoch_net_ben
        else:
            m_t = 0
            fdp_t = 0.0  # Defined as 0 when nothing migrates (Plan v3 correction 1)
            epoch_net_ben = 0.0

        epoch_fdp_list.append(fdp_t)
        epoch_migs_list.append(m_t)
        epoch_net_benefit_list.append(epoch_net_ben)
        if after_epoch is not None:
            after_epoch(int(w), int(m_t), float(fdp_t))

    total_epochs = len(epoch_windows)
    active_epochs = sum(1 for m in epoch_migs_list if m > 0)
    activity = float(active_epochs / total_epochs) if total_epochs > 0 else 0.0
    mean_fdp = float(np.mean(epoch_fdp_list)) if epoch_fdp_list else 0.0
    pooled_fmr = float(total_false_migrations / total_migrations) if total_migrations > 0 else 0.0

    return ReplayMetrics(
        total_epochs=total_epochs,
        active_epochs=active_epochs,
        migration_activity=round(activity, 4),
        total_migrations=total_migrations,
        total_false_migrations=total_false_migrations,
        mean_fdp=round(mean_fdp, 4),
        pooled_fmr=round(pooled_fmr, 4),
        total_net_benefit=round(total_net_benefit, 2),
        mean_net_benefit_per_epoch=round(total_net_benefit / total_epochs, 2) if total_epochs > 0 else 0.0,
        epoch_fdp_series=epoch_fdp_list,
        epoch_migrations=epoch_migs_list,
        epoch_net_benefits=epoch_net_benefit_list,
        decision_log=decision_log,
    )
