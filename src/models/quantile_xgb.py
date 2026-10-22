"""
Quantile XGBoost model suite for ISM.

Trains one XGBRegressor per quantile level in Q = {0.05, 0.10, ..., 0.50, 0.90}.
Predictions are on log1p(Y) scale.  After prediction, a post-hoc sort ensures
non-crossing: q05 <= q10 <= ... <= q90 per sample (no monotone-constraint refit).
The pre-sort crossing rate is reported for transparency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass
class QuantileModelSuite:
    """Container for trained quantile models and their predictions."""
    quantile_levels: List[float]
    models: Dict[float, xgb.XGBRegressor] = field(default_factory=dict)
    crossing_rate: float = 0.0  # fraction of samples with any pre-sort crossing

    def predict_raw(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """Return raw (un-sorted) predictions in original Y space."""
        preds = {}
        for q in self.quantile_levels:
            log_pred = self.models[q].predict(X)
            preds[q] = np.expm1(np.clip(log_pred, 0.0, 20.0))
        return preds

    def predict_sorted(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """
        Return non-crossing predictions: sort across quantiles per sample.
        Also computes the crossing rate.
        """
        raw = self.predict_raw(X)
        n = len(list(raw.values())[0])
        sorted_q = sorted(self.quantile_levels)

        # Stack into (n_samples, n_quantiles)
        pred_matrix = np.column_stack([raw[q] for q in sorted_q])

        # Count crossings before sort
        crossings = 0
        for i in range(pred_matrix.shape[1] - 1):
            crossings += int(np.sum(pred_matrix[:, i] > pred_matrix[:, i + 1]))
        total_pairs = n * (pred_matrix.shape[1] - 1)
        self.crossing_rate = crossings / max(1, total_pairs)

        # Sort per sample
        pred_sorted = np.sort(pred_matrix, axis=1)

        result = {}
        for idx, q in enumerate(sorted_q):
            result[q] = pred_sorted[:, idx]
        return result


def train_quantile_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    quantile_levels: List[float],
    xgb_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> QuantileModelSuite:
    """
    Train one XGBRegressor per quantile level using pinball (quantile) loss.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
    y_train : array-like of shape (n_samples,), raw Y (access counts)
    quantile_levels : list of floats in (0, 1)
    xgb_params : optional override for XGBoost hyperparameters
    verbose : print progress

    Returns
    -------
    QuantileModelSuite with trained models.
    """
    params = {
        "max_depth": 6,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "n_jobs": 2,
        "random_state": 42,
        "verbosity": 0,
    }
    if xgb_params:
        params.update(xgb_params)

    # Transform target: log1p(Y)
    y_log = np.log1p(y_train.astype(np.float64))

    suite = QuantileModelSuite(quantile_levels=sorted(quantile_levels))

    for q in sorted(quantile_levels):
        if verbose:
            print(f"  Training quantile model q={q:.2f} ...", end="", flush=True)

        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            n_jobs=params["n_jobs"],
            random_state=params["random_state"],
            verbosity=params["verbosity"],
        )
        model.fit(X_train, y_log)
        suite.models[q] = model

        if verbose:
            # Quick in-sample check
            pred_log = model.predict(X_train)
            pred_y = np.expm1(np.clip(pred_log, 0.0, 20.0))
            coverage_below = np.mean(y_train <= pred_y)
            print(f" done. In-sample coverage(Y <= q{q:.0%}) = {coverage_below:.3f}")

    return suite


def evaluate_quantile_models(
    suite: QuantileModelSuite,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate trained quantile models on held-out data.

    Returns a DataFrame with columns:
    - quantile, pinball_loss, mae_if_median, coverage_below
    Plus crossing rate info.
    """
    preds = suite.predict_sorted(X_eval)

    records = []
    for q in sorted(suite.quantile_levels):
        pred_y = preds[q]

        # Pinball (quantile) loss
        errors = y_eval - pred_y
        pinball = np.where(errors >= 0, q * errors, (q - 1) * errors)
        mean_pinball = float(np.mean(pinball))

        # MAE (useful when q=0.50)
        mae = float(np.mean(np.abs(errors)))

        # Coverage: P(Y <= q_hat)
        coverage_below = float(np.mean(y_eval <= pred_y))

        records.append({
            "quantile": q,
            "pinball_loss": round(mean_pinball, 4),
            "mae": round(mae, 4),
            "coverage_below": round(coverage_below, 4),
        })

    df = pd.DataFrame(records)
    return df


def compute_interval_metrics(
    suite: QuantileModelSuite,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    q_lo: float = 0.10,
    q_hi: float = 0.90,
) -> Dict[str, float]:
    """
    Compute prediction interval [q_lo, q_hi] coverage and width.
    """
    preds = suite.predict_sorted(X_eval)

    if q_lo not in preds or q_hi not in preds:
        raise ValueError(f"Quantile levels {q_lo} and/or {q_hi} not in trained models")

    lo = preds[q_lo]
    hi = preds[q_hi]

    in_interval = (y_eval >= lo) & (y_eval <= hi)
    coverage = float(np.mean(in_interval))
    mean_width = float(np.mean(hi - lo))
    median_width = float(np.median(hi - lo))

    return {
        "interval": f"[q{q_lo:.2f}, q{q_hi:.2f}]",
        "coverage": round(coverage, 4),
        "nominal_coverage": round(q_hi - q_lo, 4),
        "mean_width": round(mean_width, 4),
        "median_width": round(median_width, 4),
    }
