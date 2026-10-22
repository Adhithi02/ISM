"""
Metrics module for the ISM experiment.

Computes prediction, system, decision, and guarantee metrics:
  - Prediction: pinball loss, MAE, interval coverage, interval width
  - Decision: mean FMR, pooled FMR, migration activity, net benefit
  - Guarantee: violation rate, regret vs oracle
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DecisionMetrics:
    """Metrics for a single experiment run."""
    mean_fmr: float           # mean of per-epoch FDP
    pooled_fmr: float         # total wasted / total migrations
    migration_activity: float # fraction of epochs with >= 1 migration
    total_migrations: int
    wasted_migrations: int
    net_benefit: float        # sum of realized net benefits
    n_epochs: int
    n_epochs_with_mig: int


def compute_decision_metrics(
    epoch_fdps: np.ndarray,
    epoch_migrations: np.ndarray,
    epoch_wasted: np.ndarray,
    epoch_net_benefits: np.ndarray,
) -> DecisionMetrics:
    """
    Compute decision metrics from per-epoch arrays.

    Parameters
    ----------
    epoch_fdps : FDP per epoch (0 if no migrations)
    epoch_migrations : total migrations per epoch
    epoch_wasted : wasted migrations per epoch
    epoch_net_benefits : net benefit per epoch
    """
    n_epochs = len(epoch_fdps)
    has_mig = epoch_migrations > 0
    n_with_mig = int(has_mig.sum())

    mean_fmr = float(np.mean(epoch_fdps[has_mig])) if n_with_mig > 0 else 0.0
    total_mig = int(epoch_migrations.sum())
    total_wasted = int(epoch_wasted.sum())
    pooled_fmr = total_wasted / max(1, total_mig)
    activity = n_with_mig / max(1, n_epochs)
    net_benefit = float(epoch_net_benefits.sum())

    return DecisionMetrics(
        mean_fmr=round(mean_fmr, 4),
        pooled_fmr=round(pooled_fmr, 4),
        migration_activity=round(activity, 4),
        total_migrations=total_mig,
        wasted_migrations=total_wasted,
        net_benefit=round(net_benefit, 2),
        n_epochs=n_epochs,
        n_epochs_with_mig=n_with_mig,
    )


def similar_net_benefit(
    nb_method: float,
    nb_baseline: float,
    tolerance: float = 0.10,
) -> bool:
    """Check if two methods have similar net benefit (within +/- tolerance)."""
    if nb_baseline == 0:
        return abs(nb_method) < 1.0  # near-zero check
    ratio = abs(nb_method - nb_baseline) / abs(nb_baseline)
    return ratio <= tolerance


def compute_regret(
    nb_oracle: float,
    nb_method: float,
) -> float:
    """Compute regret = oracle net benefit - method net benefit."""
    return round(nb_oracle - nb_method, 2)


def evaluate_guarantee_t1(
    block_mean_fmrs: np.ndarray,
    alpha_fmr: float,
    target_fraction: float = 0.90,
) -> Dict[str, float]:
    """
    T1: Empirical mean FMR <= alpha_FMR in >= target_fraction of blocks.
    """
    n_blocks = len(block_mean_fmrs)
    n_pass = int(np.sum(block_mean_fmrs <= alpha_fmr))
    fraction = n_pass / max(1, n_blocks)

    return {
        "target": "T1",
        "criterion": f"Mean FMR <= {alpha_fmr} in >= {target_fraction*100:.0f}% of blocks",
        "pass_fraction": round(fraction, 4),
        "n_blocks": n_blocks,
        "n_pass": n_pass,
        "met": fraction >= target_fraction,
    }
