"""
Conformal Quantile Regression (CQR) — coverage analysis only.

CQR score: E_i = q_lo(X_i) - Y_i on Cal-A.
Q = ceil((n+1)(1-alpha_cov)) / n -th quantile of {E_i}.
Y_lo(x) = q_lo(x) - Q, clamped >= 0.

CQR is used for coverage reporting only; it does NOT feed into the gate decision.
The gate uses raw quantile predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class CQRResult:
    """Container for CQR calibration outputs."""
    Q_offset: float           # conformity quantile
    alpha_cov: float          # target miscoverage
    n_cal: int                # calibration set size
    cal_coverage: float       # empirical coverage on cal set
    quantile_level_used: float  # e.g. 0.10


def calibrate_cqr(
    q_lo_cal: np.ndarray,
    y_cal: np.ndarray,
    alpha_cov: float = 0.10,
) -> CQRResult:
    """
    Calibrate CQR on Cal-A data.

    Parameters
    ----------
    q_lo_cal : lower quantile predictions on Cal-A (e.g. q10)
    y_cal : true Y values on Cal-A
    alpha_cov : target miscoverage rate

    Returns
    -------
    CQRResult with the conformity offset Q.
    """
    # CQR scores: E_i = q_lo(X_i) - Y_i
    scores = q_lo_cal - y_cal
    n = len(scores)

    # Quantile index: ceil((n+1)(1-alpha)) / n
    quantile_level = np.ceil((n + 1) * (1 - alpha_cov)) / n
    quantile_level = min(quantile_level, 1.0)
    Q = float(np.quantile(scores, quantile_level))

    # Empirical coverage on cal set itself
    Y_lo_cal = np.maximum(q_lo_cal - Q, 0.0)
    cal_coverage = float(np.mean(y_cal >= Y_lo_cal))

    return CQRResult(
        Q_offset=Q,
        alpha_cov=alpha_cov,
        n_cal=n,
        cal_coverage=cal_coverage,
        quantile_level_used=quantile_level,
    )


def apply_cqr(
    q_lo: np.ndarray,
    Q_offset: float,
) -> np.ndarray:
    """
    Apply CQR conformalized lower bound: Y_lo(x) = q_lo(x) - Q, clamped >= 0.
    """
    return np.maximum(q_lo - Q_offset, 0.0)


def evaluate_cqr_coverage(
    q_lo_test: np.ndarray,
    y_test: np.ndarray,
    Q_offset: float,
) -> Dict[str, float]:
    """
    Evaluate CQR coverage on a test set.

    Returns coverage, mean/median Y_lo, and fraction of Y_lo > 0.
    """
    Y_lo = apply_cqr(q_lo_test, Q_offset)
    coverage = float(np.mean(y_test >= Y_lo))
    nonzero_frac = float(np.mean(Y_lo > 0))

    return {
        "coverage": round(coverage, 4),
        "mean_Y_lo": round(float(np.mean(Y_lo)), 4),
        "median_Y_lo": round(float(np.median(Y_lo)), 4),
        "nonzero_fraction": round(nonzero_frac, 4),
    }
