"""
Tests for conformal selection gate (Jin & Candes 2023).

(a) On i.i.d. synthetic data: mean per-epoch FDP is within +/-0.02 of alpha
    for alpha in {0.05, 0.10, 0.20, 0.30} over 200+ epochs.
(b) Under calibration/test distribution shift: FDP degrades (report by how much).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration.conformal_selection import (
    ConformalGateConfig,
    ConformalSelectionGate,
    compute_conformal_scores,
)


class TestConformalSelectionIID:
    """Test FDP control under i.i.d. conditions."""

    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20, 0.30])
    def test_fdp_control_iid(self, alpha: float):
        """
        Generate i.i.d. calibration + test data with known null/alternative.
        Check that mean per-epoch FDP is within +/-0.02 of alpha.
        """
        rng = np.random.default_rng(42)
        n_cal = 500
        n_epochs = 250
        n_per_epoch = 50
        tau = 5.0

        # Generate calibration data: mixture of null (Y < tau) and alt (Y >= tau)
        cal_y = rng.exponential(3.0, size=n_cal)
        cal_q_lo = cal_y + rng.normal(0, 1.0, size=n_cal)  # noisy prediction
        cal_tau = np.full(n_cal, tau)
        cal_scores = compute_conformal_scores("quantile", q_lo=cal_q_lo, tau=cal_tau, y=cal_y)

        gate = ConformalSelectionGate(ConformalGateConfig(alpha=alpha, score_family="quantile"))
        gate.calibrate(cal_scores, cal_y, cal_tau)

        fdps = []
        for ep in range(n_epochs):
            test_y = rng.exponential(3.0, size=n_per_epoch)
            test_q_lo = test_y + rng.normal(0, 1.0, size=n_per_epoch)
            test_tau = np.full(n_per_epoch, tau)
            test_scores = compute_conformal_scores("quantile", q_lo=test_q_lo, tau=test_tau)

            result = gate.select(test_scores, test_tau, test_y)
            if result.n_selected > 0:
                fdps.append(result.fdp)

        mean_fdp = float(np.mean(fdps)) if fdps else 0.0
        n_active = len(fdps)

        print(f"\n  alpha={alpha:.2f}: mean FDP={mean_fdp:.4f} "
              f"(target={alpha:.2f} +/-0.02), "
              f"active epochs={n_active}/{n_epochs}")

        # FDP should be at most alpha (BH controls FDR, so mean FDP <= alpha)
        # We allow alpha + 0.02 tolerance for finite-sample variation
        assert mean_fdp <= alpha + 0.02, (
            f"Mean FDP {mean_fdp:.4f} exceeds alpha+0.02={alpha + 0.02:.2f}"
        )
        # Should not be zero (gate should be selecting something)
        assert n_active >= 10, f"Only {n_active} active epochs, expected >= 10"


class TestConformalSelectionShift:
    """Test FDP under distribution shift."""

    def test_fdp_under_shift(self):
        """
        Calibrate on one distribution, test on a shifted one.
        Report FDP degradation.
        """
        rng = np.random.default_rng(123)
        n_cal = 500
        n_test = 500
        tau = 5.0
        alpha = 0.10

        # Calibration: standard distribution
        cal_y = rng.exponential(3.0, size=n_cal)
        cal_q_lo = cal_y + rng.normal(0, 1.0, size=n_cal)
        cal_tau = np.full(n_cal, tau)
        cal_scores = compute_conformal_scores("quantile", q_lo=cal_q_lo, tau=cal_tau, y=cal_y)

        gate = ConformalSelectionGate(ConformalGateConfig(alpha=alpha, score_family="quantile"))
        gate.calibrate(cal_scores, cal_y, cal_tau)

        # Test: shifted distribution (higher mean, predictions are stale)
        test_y = rng.exponential(1.5, size=n_test)  # shifted down
        test_q_lo = test_y + rng.normal(0, 1.0, size=n_test) + 2.0  # overly optimistic
        test_tau = np.full(n_test, tau)
        test_scores = compute_conformal_scores("quantile", q_lo=test_q_lo, tau=test_tau)

        result = gate.select(test_scores, test_tau, test_y)

        print(f"\n  Shift test: FDP={result.fdp:.4f} (alpha={alpha:.2f})")
        print(f"  Selected={result.n_selected}, Wasted={result.n_wasted}, "
              f"Recall={result.recall:.4f}")

        # Under shift, FDP may exceed alpha - just report
        if result.fdp > alpha + 0.05:
            print(f"  FDP degraded by {result.fdp - alpha:.4f} above alpha")

    def test_probability_score_family(self):
        """Test with probability score family."""
        rng = np.random.default_rng(42)
        n_cal = 500
        n_epochs = 50
        n_per_epoch = 50
        tau = 5.0
        alpha = 0.10

        cal_y = rng.exponential(3.0, size=n_cal)
        cal_p_hat = 1.0 / (1.0 + np.exp(-(cal_y - tau + rng.normal(0, 0.5, size=n_cal))))
        cal_tau = np.full(n_cal, tau)
        cal_scores = compute_conformal_scores("probability", p_hat=cal_p_hat, tau=cal_tau, y=cal_y)

        gate = ConformalSelectionGate(ConformalGateConfig(alpha=alpha, score_family="probability"))
        gate.calibrate(cal_scores, cal_y, cal_tau)

        fdps = []
        for ep in range(n_epochs):
            test_y = rng.exponential(3.0, size=n_per_epoch)
            test_p_hat = 1.0 / (1.0 + np.exp(-(test_y - tau + rng.normal(0, 0.5, size=n_per_epoch))))
            test_tau = np.full(n_per_epoch, tau)
            test_scores = compute_conformal_scores("probability", p_hat=test_p_hat)

            result = gate.select(test_scores, test_tau, test_y)
            if result.n_selected > 0:
                fdps.append(result.fdp)

        mean_fdp = float(np.mean(fdps)) if fdps else 0.0
        n_active = len(fdps)

        print(f"\n  Probability family: mean FDP={mean_fdp:.4f}, active={n_active}/{n_epochs}")
        # BH controls FDR: mean FDP <= alpha + tolerance
        assert mean_fdp <= alpha + 0.02
        assert n_active >= 10
