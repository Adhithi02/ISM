"""
Defer branch for uncertain migration decisions.

Extent is deferred if:
  Y_lo < tau * (1 + delta) but P_hat(Y >= tau) >= p_defer.

Deferred extents are re-evaluated at the next decision epoch.
After max_defer_epochs without qualifying -> do not migrate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DeferConfig:
    """Configuration for the defer branch."""
    p_defer: float = 0.3       # probability threshold to defer
    max_defer_epochs: int = 3  # max epochs to re-evaluate


@dataclass
class DeferDecision:
    """Decision result for an extent at a given epoch."""
    extent_id: int
    epoch: int
    action: str         # "migrate", "defer", "reject"
    p_hat: float        # calibrated probability
    Y_lo: float
    tau_threshold: float  # tau * (1 + delta)
    defer_count: int    # how many times deferred so far


class DeferTracker:
    """
    Tracks deferred decisions across epochs.

    Usage:
        tracker = DeferTracker(config)
        decisions = tracker.decide(extent_ids, Y_lo, tau_thresholds, p_hat, epoch)
        # At next epoch, call decide() again; previously deferred extents are
        # automatically re-evaluated.
    """

    def __init__(self, config: Optional[DeferConfig] = None) -> None:
        self.cfg = config or DeferConfig()
        # Maps extent_id -> defer_count
        self._deferred: Dict[int, int] = {}
        self._history: List[DeferDecision] = []

    def decide(
        self,
        extent_ids: np.ndarray,
        Y_lo: np.ndarray,
        tau_thresholds: np.ndarray,
        p_hat: np.ndarray,
        epoch: int,
    ) -> List[DeferDecision]:
        """
        Make migration decisions for a batch of extents at a given epoch.

        Parameters
        ----------
        extent_ids : array of extent identifiers
        Y_lo : conformalized lower bounds (from the gate's quantile level)
        tau_thresholds : tau(x) * (1 + delta) per extent
        p_hat : calibrated P(Y >= tau) per extent
        epoch : current decision epoch

        Returns
        -------
        List of DeferDecision objects.
        """
        decisions = []
        new_deferred = {}

        for i in range(len(extent_ids)):
            eid = int(extent_ids[i])
            y_lo = float(Y_lo[i])
            thresh = float(tau_thresholds[i])
            p = float(p_hat[i])
            defer_count = self._deferred.get(eid, 0)

            if y_lo >= thresh:
                # Passes gate -> migrate
                action = "migrate"
            elif p >= self.cfg.p_defer and defer_count < self.cfg.max_defer_epochs:
                # High probability but below gate threshold -> defer
                action = "defer"
                new_deferred[eid] = defer_count + 1
            else:
                # Low probability or max defers reached -> reject
                action = "reject"

            dec = DeferDecision(
                extent_id=eid,
                epoch=epoch,
                action=action,
                p_hat=round(p, 4),
                Y_lo=round(y_lo, 4),
                tau_threshold=round(thresh, 4),
                defer_count=defer_count,
            )
            decisions.append(dec)
            self._history.append(dec)

        # Update deferred set
        self._deferred = new_deferred
        return decisions

    def get_summary(self) -> Dict[str, int]:
        """Return summary counts of all decisions made."""
        migrate = sum(1 for d in self._history if d.action == "migrate")
        defer = sum(1 for d in self._history if d.action == "defer")
        reject = sum(1 for d in self._history if d.action == "reject")
        return {
            "migrate": migrate,
            "defer": defer,
            "reject": reject,
            "total": len(self._history),
            "currently_deferred": len(self._deferred),
        }

    @property
    def history(self) -> List[DeferDecision]:
        return self._history
