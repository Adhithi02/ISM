"""
LTT validity tests:
  Test A (i.i.d.): 50 synthetic blocks → assert violation rate <= 15%.
  Test B (autocorrelated, AR(1) rho=0.5): report violation rate, no assertion.
  Test C (distribution shift at midpoint): report violation rate, no assertion.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration.ltt_gate import _hoeffding_bentkus_pvalue


class TestLTTValidity:
    """Validate LTT gate under various data conditions."""

    def _run_synthetic_ltt(
        self,
        n_blocks: int = 50,
        n_samples: int = 200,
        alpha_fmr: float = 0.10,
        delta_ltt: float = 0.10,
        noise_type: str = "iid",
    ) -> float:
        """
        Generate synthetic FMR data and test whether the LTT gate
        controls violations at the target rate.

        Returns violation rate across blocks.
        """
        rng = np.random.default_rng(42)
        violations = 0

        for block in range(n_blocks):
            if noise_type == "iid":
                # i.i.d. FMR samples from Beta distribution centered around alpha_fmr
                fdps = rng.beta(2, 18, size=n_samples)  # mean ~0.10

            elif noise_type == "ar1":
                # AR(1) autocorrelated FMR with rho=0.5
                fdps = np.zeros(n_samples)
                fdps[0] = rng.beta(2, 18)
                rho = 0.5
                for t in range(1, n_samples):
                    innovation = rng.beta(2, 18)
                    fdps[t] = rho * fdps[t - 1] + (1 - rho) * innovation

            elif noise_type == "drift":
                # Distribution shift at midpoint
                half = n_samples // 2
                fdps_1 = rng.beta(2, 18, size=half)    # mean ~0.10
                fdps_2 = rng.beta(4, 6, size=n_samples - half)  # mean ~0.40 (drift)
                fdps = np.concatenate([fdps_1, fdps_2])

            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")

            # Compute empirical mean FDP (our test statistic)
            mean_fdp = float(np.mean(fdps))

            # H-B p-value
            p_val = _hoeffding_bentkus_pvalue(mean_fdp, alpha_fmr, n_samples)

            # Check if we'd certify (reject H0: FMR > alpha)
            # A "violation" means the true FMR exceeds alpha but we certified
            if p_val <= delta_ltt and mean_fdp > alpha_fmr:
                violations += 1

        return violations / n_blocks

    def test_fmr_guarantee_iid(self):
        """i.i.d. test: violation rate should be <= 15%."""
        viol_rate = self._run_synthetic_ltt(
            n_blocks=50, n_samples=200, noise_type="iid"
        )
        print(f"\n  LTT i.i.d. violation rate: {viol_rate:.3f} (target <= 0.15)")
        assert viol_rate <= 0.15, (
            f"i.i.d. LTT violation rate {viol_rate:.3f} exceeds 0.15"
        )

    def test_fmr_autocorrelated(self):
        """AR(1) ρ=0.5 test: report violation rate (no assertion)."""
        viol_rate = self._run_synthetic_ltt(
            n_blocks=50, n_samples=200, noise_type="ar1"
        )
        print(f"\n  LTT AR(1) violation rate: {viol_rate:.3f} (report only)")

    def test_fmr_drifted(self):
        """Distribution shift test: report violation rate (no assertion)."""
        viol_rate = self._run_synthetic_ltt(
            n_blocks=50, n_samples=200, noise_type="drift"
        )
        print(f"\n  LTT drifted violation rate: {viol_rate:.3f} (report only)")
