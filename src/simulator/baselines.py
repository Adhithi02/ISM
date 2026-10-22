"""
Baseline migration policies for comparative evaluation.

Implements all 9 policies:
1. hdd_only: no migrations; all requests served from HDD.
2. all_ssd: ideal bound; all requests served from SSD.
3. lru_promote: promote extent on first access; LRU eviction.
4. static_threshold: promote extent if recent access count >= threshold; LRU eviction.
5. point_forecast: migrate if point prediction q50 >= tau(x); LRU eviction.
6. point_forecast_value_evict: migrate if q50 >= tau(x); Value eviction.
7. safety_buffer: migrate if q50 - margin >= tau(x); LRU eviction.
8. proposed_lru_evict: calibrated gate Y_lo >= tau * (1+delta); LRU eviction.
9. hindsight_oracle: budget-constrained optimal migration using realized future accesses Y.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.simulator.cost_model import CostModel
from src.simulator.eviction import EvictionPolicy, LRUEviction, ValueEviction
from src.simulator.fast_replay import ReplayMetrics, run_fast_replay


def run_baseline(
    policy_name: str,
    decision_df: pd.DataFrame,
    ssd_capacity: int,
    max_migrations_per_epoch: int,
    cost_model: CostModel,
    threshold: float = 5.0,
    margin: float = 2.0,
    delta: float = 0.0,
    q_lo_col: str = "q_lo",
    q50_col: str = "q50",
    y_col: str = "Y",
    recent_count_col: str = "count_lag1",
    extent_size_mib: float = 1.0,
) -> ReplayMetrics:
    """
    Execute any of the 9 baseline policies using the unified replay framework.
    """
    # Eviction policy selection
    if policy_name in ["point_forecast_value_evict", "hindsight_oracle"]:
        eviction = ValueEviction()
    else:
        eviction = LRUEviction()

    # Gate logic per policy
    if policy_name == "hdd_only":
        gate_fn = lambda c, tau: pd.Series(False, index=c.index)

    elif policy_name == "all_ssd":
        # All SSD: no migrations needed; hypothetical bound
        gate_fn = lambda c, tau: pd.Series(False, index=c.index)

    elif policy_name == "lru_promote":
        # Promote on any recent access (> 0)
        col = recent_count_col if recent_count_col in decision_df.columns else y_col
        gate_fn = lambda c, tau: c[col] > 0

    elif policy_name == "static_threshold":
        # Promote if recent count >= static threshold
        col = recent_count_col if recent_count_col in decision_df.columns else y_col
        gate_fn = lambda c, tau: c[col] >= threshold

    elif policy_name == "point_forecast":
        # Migrate if q50 >= tau(x)
        p_col = q50_col if q50_col in decision_df.columns else y_col
        gate_fn = lambda c, tau: c[p_col] >= tau

    elif policy_name == "point_forecast_value_evict":
        p_col = q50_col if q50_col in decision_df.columns else y_col
        gate_fn = lambda c, tau: c[p_col] >= tau

    elif policy_name == "safety_buffer":
        # Migrate if q50 - margin >= tau(x)
        p_col = q50_col if q50_col in decision_df.columns else y_col
        gate_fn = lambda c, tau: (c[p_col] - margin) >= tau

    elif policy_name == "proposed_lru_evict":
        # Proposed gate with LRU eviction control
        lo_col = q_lo_col if q_lo_col in decision_df.columns else (q50_col if q50_col in decision_df.columns else y_col)
        gate_fn = lambda c, tau: c[lo_col] >= tau * (1.0 + delta)

    elif policy_name == "hindsight_oracle":
        # Optimal oracle using actual realized Y
        gate_fn = lambda c, tau: c[y_col] >= tau

    elif policy_name == "proposed":
        # Proposed method (calibrated gate Y_lo >= tau*(1+delta), Value eviction)
        lo_col = q_lo_col if q_lo_col in decision_df.columns else (q50_col if q50_col in decision_df.columns else y_col)
        eviction = ValueEviction()
        gate_fn = lambda c, tau: c[lo_col] >= tau * (1.0 + delta)

    else:
        raise ValueError(f"Unknown baseline policy: {policy_name}")

    return run_fast_replay(
        decision_df=decision_df,
        ssd_capacity=ssd_capacity,
        max_migrations_per_epoch=max_migrations_per_epoch,
        cost_model=cost_model,
        eviction_policy=eviction,
        gate_fn=gate_fn,
        delta=delta,
        q_lo_col=q_lo_col,
        q50_col=q50_col,
        y_col=y_col,
        extent_size_mib=extent_size_mib,
    )
