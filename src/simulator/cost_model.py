"""
Cost model and state-dependent break-even threshold computation.

Break-even threshold:
    tau(x) = (C_mig + V_evict(x) + C_wb) / Delta_L

Where:
    - Delta_L = L_hdd - L_ssd (latency saving per foreground hit on SSD)
    - C_mig = L_hdd + L_ssd (migration read from HDD + write to SSD)
    - C_wb = dirty_fraction * L_hdd (writeback cost for displaced dirty blocks)
    - V_evict(x) = predicted_q50(evictee) * Delta_L (value of displaced extent)
      (V_evict = 0 if SSD has free space).

Rule: "Decide with prediction (q50), label with realization (actual future counts)."
Rescaling: If median(tau) < 5.0, C_mig is scaled upward so median(tau) >= 5.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CostModelConfig:
    l_hdd_ms: float = 8.0
    l_ssd_ms: float = 0.1
    dirty_fraction: float = 0.3
    c_mig_scale: float = 1.0
    min_median_tau: float = 5.0
    extent_size_mib: float = 1.0
    kappa_interference: float = 0.0
    measured_utilization: float = 0.0
    ssd_wear_cost_per_mib: float = 0.0


class CostModel:
    """Computes state-dependent break-even migration thresholds."""

    def __init__(self, config: Optional[CostModelConfig] = None) -> None:
        self.cfg = config or CostModelConfig()
        self.delta_l = max(0.01, self.cfg.l_hdd_ms - self.cfg.l_ssd_ms)
        size = max(0.001, self.cfg.extent_size_mib)
        interference = 1.0 + max(0.0, self.cfg.kappa_interference) * max(
            0.0, min(1.0, self.cfg.measured_utilization)
        )
        self.migration_read_write_cost = (
            (self.cfg.l_hdd_ms + self.cfg.l_ssd_ms) * size * interference
        )
        self.base_c_mig = self.migration_read_write_cost * self.cfg.c_mig_scale
        self.writeback_cost = self.cfg.dirty_fraction * self.cfg.l_hdd_ms * size
        self.wear_cost = self.cfg.ssd_wear_cost_per_mib * size
        self.c_wb = self.writeback_cost + self.wear_cost
        self.rescale_factor: float = 1.0

    def cost_breakdown(self) -> Dict[str, float]:
        """Return physically interpretable migration-cost components."""
        return {
            "extent_size_mib": float(self.cfg.extent_size_mib),
            "migration_read_write": float(self.migration_read_write_cost),
            "writeback": float(self.writeback_cost),
            "ssd_wear": float(self.wear_cost),
            "interference_multiplier": float(
                1.0 + max(0.0, self.cfg.kappa_interference)
                * max(0.0, min(1.0, self.cfg.measured_utilization))
            ),
            "total_fixed_cost": float(self.base_c_mig + self.c_wb),
        }

    def compute_tau(
        self,
        evictee_pred_q50: Optional[float] = None,
    ) -> float:
        """
        Compute break-even access threshold tau(x) for deciding migration.
        If evictee_pred_q50 is None or 0, no extent is displaced (V_evict = 0).
        """
        v_evict = (max(0.0, evictee_pred_q50) * self.delta_l) if evictee_pred_q50 is not None else 0.0
        total_cost = (self.base_c_mig * self.rescale_factor) + v_evict + self.c_wb
        return float(total_cost / self.delta_l)

    def compute_realized_net_benefit(
        self,
        migrated_realized_y: float,
        evictee_realized_y: Optional[float] = None,
    ) -> float:
        """
        Compute realized net benefit after the epoch:
        Benefit = (Y_migrated * Delta_L) - C_mig - C_wb - (Y_evictee * Delta_L).
        """
        v_lost = (max(0.0, evictee_realized_y) * self.delta_l) if evictee_realized_y is not None else 0.0
        benefit = (migrated_realized_y * self.delta_l) - (self.base_c_mig * self.rescale_factor) - self.c_wb - v_lost
        return float(benefit)

    def is_migration_useful(
        self,
        migrated_realized_y: float,
        evictee_realized_y: Optional[float] = None,
    ) -> bool:
        """Label migration as useful iff realized net benefit > 0."""
        return self.compute_realized_net_benefit(migrated_realized_y, evictee_realized_y) > 0.0

    def check_and_rescale_tau_distribution(
        self,
        sample_evictee_q50s: List[float],
    ) -> Dict[str, float]:
        """
        Evaluate tau distribution across sample evictee predictions.
        If median(tau) < min_median_tau, rescale base_c_mig so median(tau) >= min_median_tau.
        """
        raw_taus = [self.compute_tau(q) for q in sample_evictee_q50s]
        raw_median = float(np.median(raw_taus)) if raw_taus else self.compute_tau(None)

        if raw_median < self.cfg.min_median_tau:
            # We want: median((base_c_mig * r + V_evict + C_wb) / Delta_L) = min_median_tau
            # base_c_mig * r / Delta_L + median((V_evict + C_wb) / Delta_L) = min_median_tau
            other_median = float(np.median([(max(0.0, q) * self.delta_l + self.c_wb) / self.delta_l for q in sample_evictee_q50s]))
            needed_mig_tau = max(0.5, self.cfg.min_median_tau - other_median)
            self.rescale_factor = float((needed_mig_tau * self.delta_l) / self.base_c_mig)

            adjusted_taus = [self.compute_tau(q) for q in sample_evictee_q50s]
            adj_median = float(np.median(adjusted_taus))
        else:
            adj_median = raw_median

        return {
            "raw_median_tau": round(raw_median, 2),
            "rescale_factor": round(self.rescale_factor, 3),
            "final_median_tau": round(adj_median, 2),
            "p10": round(float(np.percentile(raw_taus, 10)), 2) if raw_taus else 0.0,
            "p90": round(float(np.percentile(raw_taus, 90)), 2) if raw_taus else 0.0,
        }
