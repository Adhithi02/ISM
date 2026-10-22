"""
Online tracker for adaptive threshold adjustment.

Additive update: delta_{t+1} = delta_t + eta * (FMR_t - alpha_FMR)
  - Clamped to [delta_min, delta_max]
  - Dimensionless eta (default 0.5)
  - Delayed feedback: labels known only after h windows
  - Starts at the certified delta from LTT

The tracker is a heuristic — no formal guarantee; validated empirically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TrackerConfig:
    """Configuration for the online tracker."""
    eta: float = 0.5          # dimensionless step size
    delta_min: float = -0.5
    delta_max: float = 2.0
    h: int = 5                # feedback delay in windows


@dataclass
class TrackerState:
    """Snapshot of tracker state at a given time."""
    epoch: int
    delta: float
    fmr_observed: float
    alpha_target: float
    update_applied: float     # eta * (FMR - alpha)


class OnlineTracker:
    """
    Online additive tracker for delta adjustment.

    Usage:
        tracker = OnlineTracker(config, initial_delta=certified_delta, alpha_fmr=0.10)
        for epoch in epochs:
            current_delta = tracker.current_delta
            # ... run gate with current_delta ...
            # After h windows, when labels are available:
            tracker.update(epoch, observed_fmr)
    """

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        initial_delta: float = 0.0,
        alpha_fmr: float = 0.10,
    ) -> None:
        self.cfg = config or TrackerConfig()
        self._delta = np.clip(initial_delta, self.cfg.delta_min, self.cfg.delta_max)
        self.alpha_fmr = alpha_fmr
        self._history: List[TrackerState] = []
        self._pending_epochs: List[int] = []  # epochs awaiting feedback

    @property
    def current_delta(self) -> float:
        return float(self._delta)

    def register_decision(self, epoch: int) -> None:
        """Register that a decision was made at this epoch (feedback will arrive later)."""
        self._pending_epochs.append(epoch)

    def update(self, epoch: int, observed_fmr: float) -> TrackerState:
        """
        Apply additive update when feedback for an epoch becomes available.

        delta_{t+1} = clamp(delta_t + eta * (FMR_t - alpha_FMR))
        """
        step = self.cfg.eta * (observed_fmr - self.alpha_fmr)
        new_delta = self._delta + step
        new_delta = np.clip(new_delta, self.cfg.delta_min, self.cfg.delta_max)

        state = TrackerState(
            epoch=epoch,
            delta=round(float(new_delta), 6),
            fmr_observed=round(observed_fmr, 4),
            alpha_target=self.alpha_fmr,
            update_applied=round(step, 6),
        )
        self._history.append(state)
        self._delta = new_delta

        # Remove from pending
        if epoch in self._pending_epochs:
            self._pending_epochs.remove(epoch)

        return state

    def simulate_trajectory(
        self,
        epoch_fmrs: np.ndarray,
        feedback_delay: int = 0,
    ) -> List[TrackerState]:
        """
        Simulate the tracker over a sequence of epoch FMRs.

        Parameters
        ----------
        epoch_fmrs : array of FMR values per epoch
        feedback_delay : number of epochs of delay (0 = immediate)

        Returns
        -------
        List of TrackerState snapshots.
        """
        trajectory = []
        # Buffer for delayed feedback
        buffer = []

        for t, fmr in enumerate(epoch_fmrs):
            buffer.append((t, fmr))

            # Process feedback that's old enough
            if len(buffer) > feedback_delay:
                fb_epoch, fb_fmr = buffer.pop(0)
                state = self.update(fb_epoch, fb_fmr)
                trajectory.append(state)

        # Flush remaining buffer
        for fb_epoch, fb_fmr in buffer:
            state = self.update(fb_epoch, fb_fmr)
            trajectory.append(state)

        return trajectory

    def get_summary(self) -> Dict[str, float]:
        """Return summary statistics of the tracker's trajectory."""
        if not self._history:
            return {
                "current_delta": self._delta,
                "n_updates": 0,
            }

        deltas = [s.delta for s in self._history]
        fmrs = [s.fmr_observed for s in self._history]

        return {
            "current_delta": round(float(self._delta), 4),
            "n_updates": len(self._history),
            "mean_delta": round(float(np.mean(deltas)), 4),
            "std_delta": round(float(np.std(deltas)), 4),
            "mean_fmr": round(float(np.mean(fmrs)), 4),
            "fmr_violations": int(np.sum(np.array(fmrs) > self.alpha_fmr)),
        }

    @property
    def history(self) -> List[TrackerState]:
        return self._history
