"""
Tests for budgeted migration selection.
  - test_knapsack_stub: empty when budget = 0.
  - test_capacity_respected: SSD = 10 extents -> never > 10.
  - test_bandwidth_respected: max 5 migrations/epoch -> never > 5.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.selection.budget import (
    MigrationCandidate,
    greedy_knapsack,
    select_migrations_for_epoch,
)


class TestBudget:
    """Test budgeted selection constraints."""

    def test_knapsack_stub(self):
        """Empty selection when budget = 0."""
        candidates = [
            MigrationCandidate(extent_id=i, benefit=float(i + 1))
            for i in range(10)
        ]
        result = greedy_knapsack(candidates, max_migrations=0, ssd_available=10)
        assert len(result.selected) == 0
        assert result.budget_bw_used == 0

    def test_capacity_respected(self):
        """SSD = 10 extents -> never select more than 10."""
        candidates = [
            MigrationCandidate(extent_id=i, benefit=float(20 - i))
            for i in range(30)
        ]
        result = greedy_knapsack(candidates, max_migrations=50, ssd_available=10)
        assert len(result.selected) <= 10
        assert result.budget_ssd_used <= 10

    def test_bandwidth_respected(self):
        """Max 5 migrations/epoch -> never select more than 5."""
        candidates = [
            MigrationCandidate(extent_id=i, benefit=float(20 - i))
            for i in range(20)
        ]
        result = greedy_knapsack(candidates, max_migrations=5, ssd_available=100)
        assert len(result.selected) <= 5
        assert result.budget_bw_used <= 5

    def test_sorts_by_benefit(self):
        """Higher benefit candidates should be selected first."""
        candidates = [
            MigrationCandidate(extent_id=0, benefit=1.0),
            MigrationCandidate(extent_id=1, benefit=10.0),
            MigrationCandidate(extent_id=2, benefit=5.0),
        ]
        result = greedy_knapsack(candidates, max_migrations=2, ssd_available=2)
        selected_ids = [c.extent_id for c in result.selected]
        assert selected_ids == [1, 2]  # benefit 10, 5

    def test_negative_benefit_rejected(self):
        """Negative benefit candidates should not be selected."""
        candidates = [
            MigrationCandidate(extent_id=0, benefit=-5.0),
            MigrationCandidate(extent_id=1, benefit=3.0),
        ]
        result = greedy_knapsack(candidates, max_migrations=10, ssd_available=10)
        assert len(result.selected) == 1
        assert result.selected[0].extent_id == 1

    def test_select_migrations_wrapper(self):
        """Test the convenience wrapper function."""
        extent_ids = np.array([10, 20, 30, 40])
        q50 = np.array([8.0, 2.0, 6.0, 1.0])
        tau = np.array([5.0, 5.0, 5.0, 5.0])
        gate = np.array([True, True, True, False])

        result = select_migrations_for_epoch(
            extent_ids, q50, tau, gate,
            max_migrations=2, ssd_available=10,
        )
        # Only gates passing: 10 (ben=3), 20 (ben=-3), 30 (ben=1)
        # Positive benefit: 10 (3), 30 (1). Both fit in budget.
        assert len(result.selected) == 2
        selected_ids = [c.extent_id for c in result.selected]
        assert 10 in selected_ids
        assert 30 in selected_ids
