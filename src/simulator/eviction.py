"""
Pluggable SSD eviction policies for predictive migration.

Supported policies:
- LRUEviction: evicts extent with oldest last-access timestamp.
- ValueEviction: evicts extent with lowest predicted future value (q50 * Delta_L).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set


class EvictionPolicy(ABC):
    """Abstract interface for SSD cache eviction."""

    @abstractmethod
    def record_access(self, extent: int, timestamp: float) -> None:
        """Update extent access history on SSD hit or access."""
        pass

    @abstractmethod
    def add_extent(self, extent: int, timestamp: float, predicted_value: float = 0.0) -> None:
        """Add newly migrated extent into SSD."""
        pass

    @abstractmethod
    def remove_extent(self, extent: int) -> None:
        """Remove extent from SSD."""
        pass

    @abstractmethod
    def select_evictee(
        self,
        candidate_extents: Set[int],
        predicted_values: Optional[Dict[int, float]] = None,
    ) -> int:
        """Select extent to evict from candidate_extents in SSD."""
        pass


class LRUEviction(EvictionPolicy):
    """Least Recently Used (LRU) eviction."""

    def __init__(self) -> None:
        self.last_access: Dict[int, float] = {}

    def record_access(self, extent: int, timestamp: float) -> None:
        self.last_access[extent] = timestamp

    def add_extent(self, extent: int, timestamp: float, predicted_value: float = 0.0) -> None:
        self.last_access[extent] = timestamp

    def remove_extent(self, extent: int) -> None:
        self.last_access.pop(extent, None)

    def select_evictee(
        self,
        candidate_extents: Set[int],
        predicted_values: Optional[Dict[int, float]] = None,
    ) -> int:
        if not candidate_extents:
            raise ValueError("Cannot evict from empty candidate set")
        # Return extent with smallest last_access timestamp
        return min(candidate_extents, key=lambda e: self.last_access.get(e, -float("inf")))


class ValueEviction(EvictionPolicy):
    """Value-based eviction: evicts extent with lowest predicted future accesses (q50)."""

    def __init__(self) -> None:
        self.predicted_values: Dict[int, float] = {}
        self.last_access: Dict[int, float] = {}

    def record_access(self, extent: int, timestamp: float) -> None:
        self.last_access[extent] = timestamp

    def add_extent(self, extent: int, timestamp: float, predicted_value: float = 0.0) -> None:
        self.last_access[extent] = timestamp
        self.predicted_values[extent] = predicted_value

    def update_predictions(self, predictions: Dict[int, float]) -> None:
        self.predicted_values.update(predictions)

    def remove_extent(self, extent: int) -> None:
        self.last_access.pop(extent, None)
        self.predicted_values.pop(extent, None)

    def select_evictee(
        self,
        candidate_extents: Set[int],
        predicted_values: Optional[Dict[int, float]] = None,
    ) -> int:
        if not candidate_extents:
            raise ValueError("Cannot evict from empty candidate set")
        pvals = predicted_values if predicted_values is not None else self.predicted_values
        # Return extent with minimum predicted future value; tie-break by LRU
        return min(candidate_extents, key=lambda e: (pvals.get(e, 0.0), self.last_access.get(e, 0.0)))
