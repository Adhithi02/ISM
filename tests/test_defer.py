"""
Tests for the defer branch.
  - test_deferred_re_evaluated: deferred extent is re-evaluated at next epoch.
  - test_max_defer_limit: extent is rejected after max_defer_epochs.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration.defer import DeferConfig, DeferTracker


class TestDeferBranch:
    """Test defer branch behaviour."""

    def test_deferred_re_evaluated(self):
        """Deferred extent should be re-evaluated at next epoch."""
        config = DeferConfig(p_defer=0.3, max_defer_epochs=3)
        tracker = DeferTracker(config)

        # Epoch 0: extent 42 has high P_hat but low Y_lo -> defer
        extent_ids = np.array([42])
        Y_lo = np.array([1.0])          # below threshold
        tau_thresholds = np.array([5.0]) # tau * (1 + delta) = 5.0
        p_hat = np.array([0.5])          # above p_defer=0.3
        decisions = tracker.decide(extent_ids, Y_lo, tau_thresholds, p_hat, epoch=0)
        assert decisions[0].action == "defer"
        assert decisions[0].defer_count == 0

        # Epoch 1: same extent, now Y_lo is high enough -> migrate
        Y_lo_2 = np.array([6.0])  # above threshold
        decisions_2 = tracker.decide(extent_ids, Y_lo_2, tau_thresholds, p_hat, epoch=1)
        assert decisions_2[0].action == "migrate"

    def test_max_defer_limit(self):
        """Extent should be rejected after max_defer_epochs of deferral."""
        config = DeferConfig(p_defer=0.3, max_defer_epochs=2)
        tracker = DeferTracker(config)

        extent_ids = np.array([99])
        Y_lo = np.array([1.0])          # always below threshold
        tau_thresholds = np.array([5.0])
        p_hat = np.array([0.5])          # always above p_defer

        # Epoch 0: defer (count=0)
        d0 = tracker.decide(extent_ids, Y_lo, tau_thresholds, p_hat, epoch=0)
        assert d0[0].action == "defer"

        # Epoch 1: defer again (count=1)
        d1 = tracker.decide(extent_ids, Y_lo, tau_thresholds, p_hat, epoch=1)
        assert d1[0].action == "defer"

        # Epoch 2: max reached (count=2 >= max_defer_epochs=2) -> reject
        d2 = tracker.decide(extent_ids, Y_lo, tau_thresholds, p_hat, epoch=2)
        assert d2[0].action == "reject"

    def test_migrate_bypasses_defer(self):
        """Extent that passes the gate should migrate directly, not defer."""
        config = DeferConfig(p_defer=0.3, max_defer_epochs=3)
        tracker = DeferTracker(config)

        extent_ids = np.array([10])
        Y_lo = np.array([10.0])         # above threshold
        tau_thresholds = np.array([5.0])
        p_hat = np.array([0.8])

        decisions = tracker.decide(extent_ids, Y_lo, tau_thresholds, p_hat, epoch=0)
        assert decisions[0].action == "migrate"

    def test_low_prob_rejected_immediately(self):
        """Extent with low P_hat and low Y_lo should be rejected immediately."""
        config = DeferConfig(p_defer=0.3, max_defer_epochs=3)
        tracker = DeferTracker(config)

        extent_ids = np.array([20])
        Y_lo = np.array([1.0])          # below threshold
        tau_thresholds = np.array([5.0])
        p_hat = np.array([0.1])          # below p_defer=0.3

        decisions = tracker.decide(extent_ids, Y_lo, tau_thresholds, p_hat, epoch=0)
        assert decisions[0].action == "reject"

    def test_summary_counts(self):
        """Summary should correctly count actions."""
        config = DeferConfig(p_defer=0.3, max_defer_epochs=2)
        tracker = DeferTracker(config)

        # Mix of decisions
        tracker.decide(
            np.array([1, 2, 3]),
            np.array([10.0, 1.0, 1.0]),
            np.array([5.0, 5.0, 5.0]),
            np.array([0.8, 0.5, 0.1]),
            epoch=0,
        )

        summary = tracker.get_summary()
        assert summary["migrate"] == 1  # extent 1
        assert summary["defer"] == 1    # extent 2
        assert summary["reject"] == 1   # extent 3
        assert summary["total"] == 3
