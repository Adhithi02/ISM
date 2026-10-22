"""
Statistical tests for ISM experiments.

  - Paired Wilcoxon signed-rank test with >= 10 paired units (seeds).
  - Bootstrap 95% CIs (10,000 resamples by default).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats


@dataclass
class WilcoxonResult:
    """Result of a paired Wilcoxon signed-rank test."""
    statistic: float
    p_value: float
    n_pairs: int
    significant: bool  # at alpha=0.05
    effect_size: float  # r = Z / sqrt(n)


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int
    confidence: float


def paired_wilcoxon(
    method_scores: np.ndarray,
    baseline_scores: np.ndarray,
    alpha: float = 0.05,
    min_pairs: int = 10,
) -> WilcoxonResult:
    """
    Paired Wilcoxon signed-rank test.

    Parameters
    ----------
    method_scores : per-seed metric for the proposed method
    baseline_scores : per-seed metric for the baseline
    alpha : significance level
    min_pairs : minimum required paired units

    Returns
    -------
    WilcoxonResult
    """
    n = len(method_scores)
    assert n == len(baseline_scores), "Arrays must have same length"

    if n < min_pairs:
        return WilcoxonResult(
            statistic=float("nan"),
            p_value=1.0,
            n_pairs=n,
            significant=False,
            effect_size=0.0,
        )

    diffs = method_scores - baseline_scores
    # Remove zeros
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 3:
        return WilcoxonResult(
            statistic=0.0, p_value=1.0, n_pairs=n,
            significant=False, effect_size=0.0,
        )

    stat, p_val = sp_stats.wilcoxon(nonzero, alternative="two-sided")

    # Effect size r = Z / sqrt(n)
    z_score = sp_stats.norm.ppf(1 - p_val / 2) if p_val < 1.0 else 0.0
    effect = z_score / np.sqrt(len(nonzero))

    return WilcoxonResult(
        statistic=round(float(stat), 4),
        p_value=round(float(p_val), 6),
        n_pairs=n,
        significant=p_val < alpha,
        effect_size=round(float(effect), 4),
    )


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """
    Bootstrap confidence interval for the mean.

    Parameters
    ----------
    data : sample values
    n_bootstrap : number of bootstrap resamples
    confidence : confidence level (default 0.95)
    seed : random seed
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    point = float(np.mean(data))

    boot_means = np.array([
        np.mean(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha_half = (1 - confidence) / 2
    ci_lo = float(np.percentile(boot_means, alpha_half * 100))
    ci_hi = float(np.percentile(boot_means, (1 - alpha_half) * 100))

    return BootstrapCI(
        point_estimate=round(point, 6),
        ci_lower=round(ci_lo, 6),
        ci_upper=round(ci_hi, 6),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )
