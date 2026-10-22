"""
Isotonic probability model for P(Y >= tau | x).

Two-stage pipeline:
  Stage 1 (Train): Logistic regression on features [q10(x), q50(x), q90(x), spread(x)]
      where spread = q90 - q10, producing P_raw(Y >= tau | x).
  Stage 2 (Cal-A): IsotonicRegression mapping P_raw -> P_hat, ensuring calibrated
      probability estimates.

The isotonic model is used for the defer branch (P_hat >= p_defer) and for
reporting calibration quality.  It does NOT feed into the LTT gate decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbModelResult:
    """Holds the fitted probability model and its calibration metrics."""
    logistic: LogisticRegression
    isotonic: IsotonicRegression
    scaler: StandardScaler
    tau_threshold: float
    cal_metrics: Dict[str, float]


def _build_prob_features(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> np.ndarray:
    """Stack [q10, q50, q90, spread] into feature matrix."""
    spread = q90 - q10
    return np.column_stack([q10, q50, q90, spread])


def build_family2_features(
    quantile_preds: Dict[float, np.ndarray],
    tau: np.ndarray,
) -> np.ndarray:
    """Family-2 features: [q05, q10, q20, q30, q50, q90, spread, tau]."""
    q05 = quantile_preds[0.05]
    q10 = quantile_preds[0.10]
    q20 = quantile_preds[0.20]
    q30 = quantile_preds[0.30]
    q50 = quantile_preds[0.50]
    q90 = quantile_preds[0.90]
    spread = q90 - q05
    tau_arr = np.full(len(q05), float(tau[0])) if np.ndim(tau) == 0 else np.asarray(tau, dtype=float)
    return np.column_stack([q05, q10, q20, q30, q50, q90, spread, tau_arr])


def train_family2_cal_a(
    quantile_preds_cal_a: Dict[float, np.ndarray],
    y_cal_a: np.ndarray,
    tau: np.ndarray,
    verbose: bool = False,
) -> ProbModelResult:
    """
    Logistic on the first half of Cal-A, isotonic on the second half.
    Labels are 1{Y >= tau}. Uncertainty quantiles are retained as features.
    """
    X = build_family2_features(quantile_preds_cal_a, tau)
    tau_arr = np.full(len(y_cal_a), float(np.median(tau))) if np.ndim(tau) == 0 else np.asarray(tau, dtype=float)
    z = (y_cal_a >= tau_arr).astype(np.int32)
    n = len(z)
    split = max(1, n // 2)
    X_log, z_log = X[:split], z[:split]
    X_iso, z_iso = X[split:], z[split:]
    if len(np.unique(z_log)) < 2:
        z_log = z
        X_log = X
    scaler = StandardScaler()
    X_log_s = scaler.fit_transform(X_log)
    lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=42)
    lr.fit(X_log_s, z_log)
    if len(X_iso) == 0 or len(np.unique(z_iso)) < 1:
        X_iso, z_iso = X_log, z_log
    p_raw_iso = lr.predict_proba(scaler.transform(X_iso))[:, 1]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    try:
        iso.fit(p_raw_iso, z_iso)
    except ValueError:
        iso.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    p_hat = iso.predict(lr.predict_proba(scaler.transform(X))[:, 1])
    brier = float(np.mean((p_hat - z) ** 2))
    if verbose:
        print(f"    Family-2 Cal-A Brier={brier:.4f}, n_log={len(z_log)}, n_iso={len(z_iso)}")
    return ProbModelResult(
        logistic=lr,
        isotonic=iso,
        scaler=scaler,
        tau_threshold=float(np.median(tau) if np.ndim(tau) else tau),
        cal_metrics={"brier": round(brier, 4), "ece": 0.0},
    )


def predict_family2(model: ProbModelResult, quantile_preds: Dict[float, np.ndarray], tau: np.ndarray) -> np.ndarray:
    X = build_family2_features(quantile_preds, tau)
    p_raw = model.logistic.predict_proba(model.scaler.transform(X))[:, 1]
    return model.isotonic.predict(p_raw)


def train_prob_model(
    q10_train: np.ndarray,
    q50_train: np.ndarray,
    q90_train: np.ndarray,
    y_train: np.ndarray,
    tau: float,
    q10_cal_a: np.ndarray,
    q50_cal_a: np.ndarray,
    q90_cal_a: np.ndarray,
    y_cal_a: np.ndarray,
    verbose: bool = True,
) -> ProbModelResult:
    """
    Fit the two-stage probability model.

    Parameters
    ----------
    q10/q50/q90_train : predictions from quantile models on Train set
    y_train : true Y on Train set
    tau : break-even threshold (scalar for simplicity; use median tau)
    q10/q50/q90_cal_a : predictions on Cal-A set
    y_cal_a : true Y on Cal-A set

    Returns
    -------
    ProbModelResult with fitted models and calibration diagnostics.
    """
    if verbose:
        print("  Stage 1: Fitting logistic regression on Train ...")

    # Build features
    X_train = _build_prob_features(q10_train, q50_train, q90_train)
    X_cal_a = _build_prob_features(q10_cal_a, q50_cal_a, q90_cal_a)

    # Binary labels: Y >= tau
    z_train = (y_train >= tau).astype(np.int32)
    z_cal_a = (y_cal_a >= tau).astype(np.int32)

    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_a_s = scaler.transform(X_cal_a)

    # Stage 1: logistic regression
    lr = LogisticRegression(
        C=1.0,
        max_iter=500,
        solver="lbfgs",
        random_state=42,
    )
    lr.fit(X_train_s, z_train)

    p_raw_train = lr.predict_proba(X_train_s)[:, 1]
    p_raw_cal_a = lr.predict_proba(X_cal_a_s)[:, 1]

    if verbose:
        train_acc = float(np.mean((p_raw_train >= 0.5) == z_train))
        print(f"    Logistic accuracy on Train: {train_acc:.3f}")
        print(f"    Prevalence Y>=tau: Train={z_train.mean():.3f}, Cal-A={z_cal_a.mean():.3f}")

    # Stage 2: isotonic calibration on Cal-A
    if verbose:
        print("  Stage 2: Fitting IsotonicRegression on Cal-A ...")

    iso = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )
    iso.fit(p_raw_cal_a, z_cal_a)

    p_hat_cal_a = iso.predict(p_raw_cal_a)

    # Calibration metrics on Cal-A
    brier = float(np.mean((p_hat_cal_a - z_cal_a) ** 2))
    ece = _expected_calibration_error(p_hat_cal_a, z_cal_a, n_bins=10)

    if verbose:
        print(f"    Brier score (Cal-A): {brier:.4f}")
        print(f"    ECE (Cal-A, 10 bins): {ece:.4f}")

    return ProbModelResult(
        logistic=lr,
        isotonic=iso,
        scaler=scaler,
        tau_threshold=tau,
        cal_metrics={"brier": round(brier, 4), "ece": round(ece, 4)},
    )


def predict_prob(
    model: ProbModelResult,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> np.ndarray:
    """Return calibrated P_hat(Y >= tau | x)."""
    X = _build_prob_features(q10, q50, q90)
    X_s = model.scaler.transform(X)
    p_raw = model.logistic.predict_proba(X_s)[:, 1]
    p_hat = model.isotonic.predict(p_raw)
    return p_hat


def _expected_calibration_error(
    p_hat: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (p_hat >= bin_edges[i]) & (p_hat < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (p_hat == bin_edges[i + 1])
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        avg_pred = float(p_hat[mask].mean())
        avg_true = float(y_true[mask].mean())
        ece += (n_bin / n) * abs(avg_pred - avg_true)
    return float(ece)
