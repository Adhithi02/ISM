"""
Conformal Selection gate (Jin & Candes 2023, "Selection by Prediction
with Conformal p-values").

Per decision epoch compute conformal p-values for each candidate extent.

Null hypothesis H_0(i): Y_i < tau(i)   (migration would be wasted)
Alternative:            Y_i >= tau(i)   (migration is useful)

Score function s(x, y) = tau - y  (higher = more consistent with null)
For candidate i with features x_i:
    score_i = tau(i) - score_hat(x_i)   where score_hat is the predicted
                                         quantile or isotonic-probability score

Conformal p-value:
    p_i = (1 + |{j in cal : s_j >= s_i}|) / (n_cal + 1)

Then apply BH at level alpha to reject the nulls (= select for migration).

Two score families:
  - "quantile": s(x) = tau - q_lo(x)           (lower quantile prediction)
  - "probability": s(x) = 1 - P_hat(Y >= tau)  (calibrated probability)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ConformalSelectionResult:
    """Result from conformal selection on one epoch."""
    n_candidates: int
    n_selected: int       # number of rejections (= migrations)
    n_wasted: int         # false discoveries (Y < tau after migration)
    fdp: float            # n_wasted / max(n_selected, 1)
    net_benefit: float
    recall: float         # selected_and_useful / total_useful
    p_values: np.ndarray  # per-candidate conformal p-values


@dataclass
class ConformalGateConfig:
    """Configuration for the conformal selection gate."""
    alpha: float = 0.10          # BH target FDR level
    score_family: str = "quantile"  # "quantile" or "probability"
    min_cal_size: int = 20       # minimum calibration set size


class ConformalSelectionGate:
    """
    Conformal selection gate.

    Usage:
      gate = ConformalSelectionGate(config)
      gate.calibrate(cal_scores, cal_labels, cal_taus)
      result = gate.select(test_scores, test_taus, test_y_realized)
    """

    def __init__(self, config: Optional[ConformalGateConfig] = None) -> None:
        self.cfg = config or ConformalGateConfig()
        self._cal_scores: Optional[np.ndarray] = None
        self._n_cal: int = 0

    def calibrate(
        self,
        cal_scores: np.ndarray,
        cal_y: Optional[np.ndarray] = None,
        cal_tau: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Calibrate using the calibration set.

        Parameters
        ----------
        cal_scores : conformity scores for calibration data V_i = V(X_i, Y_i)
            For "quantile" family: Y - q_lo(X)
            For "probability" family: 1{Y >= tau} - P_hat(Y >= tau | X)
        cal_y : realized Y values on calibration set (optional, for diagnostics)
        cal_tau : per-extent tau on calibration set (optional, for diagnostics)
        """
        self._cal_scores = np.sort(cal_scores)
        self._n_cal = len(cal_scores)

        # Diagnostics
        diag = {
            "n_cal": self._n_cal,
            "score_min": float(np.min(cal_scores)),
            "score_median": float(np.median(cal_scores)),
            "score_max": float(np.max(cal_scores)),
        }
        if cal_y is not None and cal_tau is not None:
            null_mask = cal_y < cal_tau
            diag["n_null"] = int(null_mask.sum())
            diag["n_alternative"] = int((~null_mask).sum())

        return diag

    def _conformal_pvalues(self, test_scores: np.ndarray) -> np.ndarray:
        """
        Compute conformal p-values for test candidates (Jin & Candes 2023).

        p_j = (1 + |{i in cal : V_i < V_hat_j}|) / (n_cal + 1)

        Lower test_score (higher predicted excess over tau) -> smaller p-value ->
        stronger evidence against null (migration is useful).
        Under H0: Y_j < tau_j, p_j is super-uniform: P(p_j <= alpha) <= alpha.
        """
        assert self._cal_scores is not None, "Must call calibrate() first"
        n = self._n_cal
        ranks = np.searchsorted(self._cal_scores, test_scores, side="left")
        p_values = (1 + ranks) / (n + 1)
        return p_values

    def select(
        self,
        test_scores: np.ndarray,
        test_tau: np.ndarray,
        test_y: np.ndarray,
        cost_model: Any = None,
        max_migrations: int = 999999,
    ) -> ConformalSelectionResult:
        """
        Apply BH procedure to conformal p-values to select extents for migration.

        Parameters
        ----------
        test_scores : conformity scores for test candidates
        test_tau : per-extent break-even thresholds
        test_y : realized Y values (for evaluation only)
        cost_model : CostModel for net benefit (optional)
        max_migrations : budget limit

        Returns
        -------
        ConformalSelectionResult
        """
        n = len(test_scores)
        if n == 0:
            return ConformalSelectionResult(
                n_candidates=0, n_selected=0, n_wasted=0,
                fdp=0.0, net_benefit=0.0, recall=0.0,
                p_values=np.array([]),
            )

        p_values = self._conformal_pvalues(test_scores)

        # BH procedure
        selected_mask = self._bh_procedure(p_values, self.cfg.alpha)

        # Budget: keep only top-max_migrations by p-value
        selected_idx = np.where(selected_mask)[0]
        if len(selected_idx) > max_migrations:
            # Keep the ones with smallest p-values
            top_idx = selected_idx[np.argsort(p_values[selected_idx])[:max_migrations]]
            selected_mask = np.zeros(n, dtype=bool)
            selected_mask[top_idx] = True

        n_selected = int(selected_mask.sum())

        # Evaluate: which selections are wasted (Y < tau)?
        useful_mask = test_y >= test_tau
        wasted_mask = selected_mask & ~useful_mask
        n_wasted = int(wasted_mask.sum())
        fdp = n_wasted / max(n_selected, 1)

        # Recall: fraction of truly useful extents that were selected
        n_useful_total = int(useful_mask.sum())
        n_useful_selected = int((selected_mask & useful_mask).sum())
        recall = n_useful_selected / max(n_useful_total, 1)

        # Net benefit
        net_benefit = 0.0
        if cost_model is not None:
            for i in np.where(selected_mask)[0]:
                net_benefit += cost_model.compute_realized_net_benefit(float(test_y[i]))

        return ConformalSelectionResult(
            n_candidates=n,
            n_selected=n_selected,
            n_wasted=n_wasted,
            fdp=round(fdp, 4),
            net_benefit=round(net_benefit, 2),
            recall=round(recall, 4),
            p_values=p_values,
        )

    @staticmethod
    def _bh_procedure(p_values: np.ndarray, alpha: float) -> np.ndarray:
        """
        Benjamini-Hochberg procedure.
        Returns boolean mask of rejected hypotheses.
        """
        n = len(p_values)
        if n == 0:
            return np.array([], dtype=bool)

        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # BH thresholds: alpha * k / n for k = 1, ..., n
        thresholds = alpha * np.arange(1, n + 1) / n

        # Find the largest k where p_(k) <= alpha * k / n
        rejections = sorted_p <= thresholds
        if not rejections.any():
            return np.zeros(n, dtype=bool)

        # All hypotheses with rank <= k_max are rejected
        k_max = np.max(np.where(rejections)[0])
        rejected_mask = np.zeros(n, dtype=bool)
        rejected_mask[sorted_idx[:k_max + 1]] = True

        return rejected_mask


def compute_conformal_scores(
    score_family: str,
    q_lo: Optional[np.ndarray] = None,
    p_hat: Optional[np.ndarray] = None,
    tau: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute conformity scores for the specified family (Jin & Candes 2023).

    For calibration points (y is not None), computes V_i = V(X_i, Y_i):
      - "quantile":     V(X, Y) = Y - q_lo(X)
      - "probability":  V(X, Y) = 1{Y >= tau} - P_hat(Y >= tau | X)

    For test candidates (y is None), evaluates V at boundary of H0: Y_j < tau_j:
      - "quantile":     V_hat(X, tau) = tau - q_lo(X)
      - "probability":  V_hat(X, tau) = 0 - P_hat(Y >= tau | X) = -P_hat

    In both cases: smaller test score -> stronger evidence of being useful.
    """
    if score_family == "quantile":
        assert q_lo is not None
        if y is not None:
            return y - q_lo
        assert tau is not None
        return tau - q_lo
    elif score_family == "probability":
        assert p_hat is not None
        if y is not None:
            assert tau is not None
            return (y >= tau).astype(float) - p_hat
        return -p_hat
    else:
        raise ValueError(f"Unknown score family: {score_family}")


def run_conformal_selection_epoch(
    gate: ConformalSelectionGate,
    epoch_q_lo: np.ndarray,
    epoch_p_hat: np.ndarray,
    epoch_tau: np.ndarray,
    epoch_y: np.ndarray,
    cost_model: Any = None,
    max_migrations: int = 10,
) -> ConformalSelectionResult:
    """
    Convenience: run conformal selection for one decision epoch.
    Uses the gate's configured score family.
    """
    scores = compute_conformal_scores(
        gate.cfg.score_family,
        q_lo=epoch_q_lo,
        p_hat=epoch_p_hat,
        tau=epoch_tau,
    )
    return gate.select(scores, epoch_tau, epoch_y, cost_model, max_migrations)
