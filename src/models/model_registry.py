"""
Unified Quantile Model Suite for ISM.

Supports multiple advanced quantile regression backends:
- 'xgboost': XGBoost Regressor with reg:quantileerror
- 'lightgbm': LightGBM Regressor with objective='quantile'
- 'catboost': CatBoost Regressor with loss_function='Quantile:alpha=q'
- 'hist_gbm': Scikit-Learn HistGradientBoostingRegressor with loss='quantile'
- 'linear_qr': Scikit-Learn QuantileRegressor (L1 quantile regression)

All backends train on log1p(Y) scale and provide sorted non-crossing
predictions in raw Y space, with empirical crossing rate tracking.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


SUPPORTED_ALGORITHMS = ["xgboost", "lightgbm", "catboost", "hist_gbm", "linear_qr"]


@dataclass
class QuantileModelSuite:
    """Container for trained quantile models and their predictions across algorithms."""
    algorithm: str
    quantile_levels: List[float]
    models: Dict[float, Any] = field(default_factory=dict)
    crossing_rate: float = 0.0  # fraction of samples with pre-sort quantile crossings
    training_time_s: float = 0.0  # total wall-clock training time in seconds

    def predict_raw(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """Return raw (un-sorted) predictions in original Y space."""
        preds = {}
        for q in self.quantile_levels:
            model = self.models[q]
            log_pred = model.predict(X)
            # Clip log predictions to avoid numeric overflow in expm1
            log_pred_clipped = np.clip(np.asarray(log_pred, dtype=np.float64), 0.0, 20.0)
            preds[q] = np.expm1(log_pred_clipped)
        return preds

    def predict_sorted(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """
        Return non-crossing predictions: sort across quantiles per sample.
        Also tracks the crossing rate before sorting for model diagnostic transparency.
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

        # Sort per sample to enforce monotonicity (q_i <= q_{i+1})
        pred_sorted = np.sort(pred_matrix, axis=1)

        result = {}
        for idx, q in enumerate(sorted_q):
            result[q] = pred_sorted[:, idx]
        return result


def train_single_quantile_model(
    algorithm: str,
    q: float,
    X_train: np.ndarray,
    y_log: np.ndarray,
    hyperparams: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Any:
    """Train a single model for quantile level q with the specified algorithm."""
    hp = hyperparams or {}

    if algorithm == "xgboost":
        default_hp = {
            "max_depth": 6,
            "n_estimators": 100,
            "learning_rate": 0.05,
            "n_jobs": 2,
            "subsample": 0.8,
            "verbosity": 0,
        }
        default_hp.update(hp)
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            max_depth=default_hp["max_depth"],
            n_estimators=default_hp["n_estimators"],
            learning_rate=default_hp["learning_rate"],
            n_jobs=default_hp["n_jobs"],
            subsample=default_hp.get("subsample", 0.8),
            random_state=random_state,
            verbosity=default_hp["verbosity"],
        )
        model.fit(X_train, y_log)
        return model

    elif algorithm == "lightgbm":
        default_hp = {
            "max_depth": 6,
            "num_leaves": 31,
            "n_estimators": 100,
            "learning_rate": 0.05,
            "n_jobs": 2,
            "subsample": 0.8,
            "subsample_freq": 1,
            "verbose": -1,
        }
        default_hp.update(hp)
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            max_depth=default_hp["max_depth"],
            num_leaves=default_hp["num_leaves"],
            n_estimators=default_hp["n_estimators"],
            learning_rate=default_hp["learning_rate"],
            n_jobs=default_hp["n_jobs"],
            subsample=default_hp.get("subsample", 0.8),
            subsample_freq=default_hp.get("subsample_freq", 1),
            random_state=random_state,
            verbose=default_hp["verbose"],
        )
        model.fit(X_train, y_log)
        return model

    elif algorithm == "catboost":
        default_hp = {
            "depth": 6,
            "iterations": 100,
            "learning_rate": 0.05,
            "thread_count": 2,
            "verbose": 0,
        }
        default_hp.update(hp)
        model = CatBoostRegressor(
            loss_function=f"Quantile:alpha={q}",
            depth=default_hp["depth"],
            iterations=default_hp["iterations"],
            learning_rate=default_hp["learning_rate"],
            thread_count=default_hp["thread_count"],
            random_seed=random_state,
            verbose=default_hp["verbose"],
        )
        model.fit(X_train, y_log)
        return model

    elif algorithm == "hist_gbm":
        default_hp = {
            "max_depth": 6,
            "max_iter": 100,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
        }
        default_hp.update(hp)
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_depth=default_hp["max_depth"],
            max_iter=default_hp["max_iter"],
            learning_rate=default_hp["learning_rate"],
            max_leaf_nodes=default_hp["max_leaf_nodes"],
            random_state=random_state,
        )
        model.fit(X_train, y_log)
        return model

    elif algorithm == "linear_qr":
        # Linear L1 Quantile Regression (subsample if dataset is large for efficiency)
        default_hp = {"alpha": 0.01, "solver": "highs"}
        default_hp.update(hp)
        n_samples = len(X_train)
        if n_samples > 5000:
            rng = np.random.default_rng(random_state)
            sub_idx = rng.choice(n_samples, size=5000, replace=False)
            X_fit = X_train[sub_idx]
            y_fit = y_log[sub_idx]
        else:
            X_fit = X_train
            y_fit = y_log
        model = QuantileRegressor(
            quantile=q,
            alpha=default_hp["alpha"],
            solver=default_hp["solver"],
        )
        model.fit(X_fit, y_fit)
        return model

    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Must be one of {SUPPORTED_ALGORITHMS}")


def train_quantile_model_suite(
    algorithm: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    quantile_levels: List[float],
    hyperparams: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    verbose: bool = False,
) -> QuantileModelSuite:
    """
    Train a complete QuantileModelSuite using the specified algorithm.

    Parameters
    ----------
    algorithm : one of ['xgboost', 'lightgbm', 'catboost', 'hist_gbm', 'linear_qr']
    X_train : feature array (n_samples, n_features)
    y_train : raw target access counts (n_samples,)
    quantile_levels : list of target quantiles (e.g. [0.05, 0.10, 0.20, 0.30, 0.50, 0.90])
    hyperparams : optional dict of hyperparameters for the algorithm
    random_state : seed for reproducibility
    verbose : whether to log progress

    Returns
    -------
    QuantileModelSuite with trained models per quantile level.
    """
    algorithm = algorithm.lower().strip()
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {SUPPORTED_ALGORITHMS}")

    y_log = np.log1p(np.asarray(y_train, dtype=np.float64))
    suite = QuantileModelSuite(
        algorithm=algorithm,
        quantile_levels=sorted(quantile_levels),
    )

    t0 = time.perf_counter()
    for q in sorted(quantile_levels):
        if verbose:
            print(f"  [{algorithm.upper()}] Training quantile model q={q:.2f} ...", end="", flush=True)
        m = train_single_quantile_model(
            algorithm=algorithm,
            q=q,
            X_train=X_train,
            y_log=y_log,
            hyperparams=hyperparams,
            random_state=random_state,
        )
        suite.models[q] = m
        if verbose:
            print(" done.")

    suite.training_time_s = round(time.perf_counter() - t0, 3)
    return suite


def evaluate_quantile_models(
    suite: QuantileModelSuite,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate trained quantile models on held-out test data.

    Returns DataFrame with:
    - quantile, pinball_loss, mae, coverage_below
    """
    preds = suite.predict_sorted(X_eval)
    y_eval_arr = np.asarray(y_eval, dtype=np.float64)

    records = []
    for q in sorted(suite.quantile_levels):
        pred_y = preds[q]
        errors = y_eval_arr - pred_y
        pinball = np.where(errors >= 0, q * errors, (q - 1) * errors)
        mean_pinball = float(np.mean(pinball))
        mae = float(np.mean(np.abs(errors)))
        coverage_below = float(np.mean(y_eval_arr <= pred_y))

        records.append({
            "quantile": q,
            "pinball_loss": round(mean_pinball, 4),
            "mae": round(mae, 4),
            "coverage_below": round(coverage_below, 4),
        })

    return pd.DataFrame(records)


def compute_interval_metrics(
    suite: QuantileModelSuite,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    q_lo: float = 0.10,
    q_hi: float = 0.90,
) -> Dict[str, float]:
    """Compute prediction interval [q_lo, q_hi] coverage, mean width, and median width."""
    preds = suite.predict_sorted(X_eval)
    if q_lo not in preds or q_hi not in preds:
        raise ValueError(f"Quantiles {q_lo} and/or {q_hi} not in trained suite: {suite.quantile_levels}")

    lo = preds[q_lo]
    hi = preds[q_hi]
    y_eval_arr = np.asarray(y_eval, dtype=np.float64)

    in_interval = (y_eval_arr >= lo) & (y_eval_arr <= hi)
    coverage = float(np.mean(in_interval))
    width = hi - lo

    return {
        "interval": f"[q{q_lo:.2f}, q{q_hi:.2f}]",
        "coverage": round(coverage, 4),
        "nominal_coverage": round(q_hi - q_lo, 4),
        "mean_width": round(float(np.mean(width)), 4),
        "median_width": round(float(np.median(width)), 4),
    }
