"""
Budgeted migration selection via greedy knapsack.

At each decision epoch, select which qualified extents to actually migrate
subject to:
  - SSD capacity constraint (total migrated + existing <= ssd_capacity)
  - Bandwidth budget (max_migrations_per_epoch)

Benefit = q50(x) - tau(x); cost = 1 bandwidth unit + 1 MiB SSD.
Greedy: sort by benefit/cost ratio descending, fill until budget exhausted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MigrationCandidate:
    """A candidate extent for migration."""
    extent_id: int
    benefit: float       # q50(x) - tau(x)
    cost_bw: int = 1     # 1 bandwidth unit
    cost_ssd: int = 1    # 1 MiB SSD space


@dataclass
class SelectionResult:
    """Result of budgeted selection for one epoch."""
    selected: List[MigrationCandidate]
    rejected: List[MigrationCandidate]
    total_benefit: float
    budget_bw_used: int
    budget_ssd_used: int
    budget_bw_limit: int
    budget_ssd_limit: int


def greedy_knapsack(
    candidates: List[MigrationCandidate],
    max_migrations: int,
    ssd_available: int,
) -> SelectionResult:
    """
    Greedy knapsack selection for migration candidates.

    Parameters
    ----------
    candidates : list of MigrationCandidate (only those passing the gate)
    max_migrations : maximum migrations per epoch (bandwidth budget)
    ssd_available : available SSD slots

    Returns
    -------
    SelectionResult with selected and rejected candidates.
    """
    # Filter to positive benefit only
    positive = [c for c in candidates if c.benefit > 0]

    # Sort by benefit ratio (benefit / cost) descending
    positive.sort(key=lambda c: c.benefit / max(c.cost_bw, 1), reverse=True)

    selected = []
    rejected = []
    bw_used = 0
    ssd_used = 0
    total_benefit = 0.0

    for c in positive:
        if bw_used + c.cost_bw > max_migrations:
            rejected.append(c)
            continue
        if ssd_used + c.cost_ssd > ssd_available:
            rejected.append(c)
            continue

        selected.append(c)
        bw_used += c.cost_bw
        ssd_used += c.cost_ssd
        total_benefit += c.benefit

    # Remaining negative-benefit candidates
    neg = [c for c in candidates if c.benefit <= 0]
    rejected.extend(neg)

    return SelectionResult(
        selected=selected,
        rejected=rejected,
        total_benefit=round(total_benefit, 4),
        budget_bw_used=bw_used,
        budget_ssd_used=ssd_used,
        budget_bw_limit=max_migrations,
        budget_ssd_limit=ssd_available,
    )


def select_migrations_for_epoch(
    extent_ids: np.ndarray,
    q50_preds: np.ndarray,
    tau_values: np.ndarray,
    gate_mask: np.ndarray,
    max_migrations: int,
    ssd_available: int,
) -> SelectionResult:
    """
    Convenience wrapper: build candidates from arrays and run greedy selection.

    Parameters
    ----------
    extent_ids : array of extent IDs
    q50_preds : q50 predictions per extent
    tau_values : break-even thresholds per extent
    gate_mask : boolean mask of extents passing the gate
    max_migrations : bandwidth budget
    ssd_available : SSD slots available

    Returns
    -------
    SelectionResult
    """
    candidates = []
    for i in range(len(extent_ids)):
        if gate_mask[i]:
            candidates.append(MigrationCandidate(
                extent_id=int(extent_ids[i]),
                benefit=float(q50_preds[i] - tau_values[i]),
            ))

    return greedy_knapsack(candidates, max_migrations, ssd_available)
