"""
Visualization module for ISM experiments.

Creates all required figures:
  - Coverage calibration plot (CQR)
  - Price-of-safety curve (net benefit vs alpha_FMR)
  - FMR over time under drift (static vs online)
  - Latency CDFs
  - Baseline comparison bars
  - LTT validity bars
  - Migration activity plot
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Consistent style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.edgecolor": "#dee2e6",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

COLORS = ["#2b5c8f", "#e040fb", "#00bcd4", "#ff9800", "#4caf50",
          "#f44336", "#9c27b0", "#795548", "#607d8b", "#cddc39"]


def plot_coverage_calibration(
    quantile_levels: List[float],
    empirical_coverages: List[float],
    save_path: Path,
    title: str = "Quantile Coverage Calibration",
) -> None:
    """Plot nominal vs empirical coverage for each quantile level."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "--", color="#aaa", label="Perfect calibration")
    ax.scatter(quantile_levels, empirical_coverages,
               c=COLORS[0], s=60, zorder=5, label="Observed")
    ax.set_xlabel("Nominal quantile level")
    ax.set_ylabel("Empirical P(Y <= q_hat)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_fmr_over_time(
    epochs: np.ndarray,
    fmr_static: np.ndarray,
    fmr_online: np.ndarray,
    alpha_fmr: float,
    save_path: Path,
    title: str = "FMR Over Time: Static vs Online Tracker",
) -> None:
    """Plot FMR trajectory for static gate vs online tracker."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, fmr_static, color=COLORS[0], alpha=0.7, label="Static gate")
    ax.plot(epochs, fmr_online, color=COLORS[1], alpha=0.7, label="Online tracker")
    ax.axhline(alpha_fmr, color="#f44336", linestyle="--", linewidth=1.5,
               label=f"alpha_FMR = {alpha_fmr}")
    ax.set_xlabel("Decision epoch")
    ax.set_ylabel("FMR (False Migration Rate)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_baseline_bars(
    policy_names: List[str],
    net_benefits: List[float],
    fmrs: List[float],
    save_path: Path,
    title: str = "Baseline Comparison",
) -> None:
    """Bar chart comparing net benefit and FMR across policies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(policy_names))
    ax1.barh(x, net_benefits, color=COLORS[0], alpha=0.8)
    ax1.set_yticks(x)
    ax1.set_yticklabels(policy_names, fontsize=8)
    ax1.set_xlabel("Net Benefit (ms)")
    ax1.set_title("Net Benefit by Policy")

    ax2.barh(x, fmrs, color=COLORS[1], alpha=0.8)
    ax2.set_yticks(x)
    ax2.set_yticklabels(policy_names, fontsize=8)
    ax2.set_xlabel("Pooled FMR")
    ax2.set_title("False Migration Rate by Policy")

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_price_of_safety(
    alphas: List[float],
    net_benefits: List[float],
    save_path: Path,
    title: str = "Price of Safety: Net Benefit vs alpha_FMR",
) -> None:
    """Line plot: net benefit as a function of alpha_FMR."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, net_benefits, "o-", color=COLORS[0], linewidth=2, markersize=8)
    ax.set_xlabel("alpha_FMR (target false-migration rate)")
    ax.set_ylabel("Net Benefit (ms)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_workload_summary(df, meta, plots_dir: Path) -> None:
    """Minimal S1 workload snapshot."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if "count" in df.columns:
        ax.plot(df.groupby("window")["count"].sum().values, color=COLORS[0])
    ax.set_xlabel("window")
    ax.set_ylabel("access count")
    ax.set_title(str(meta.get("workload_type", "workload")))
    plt.tight_layout()
    plt.savefig(plots_dir / f"{meta.get('workload_type', 'workload')}_summary.png", dpi=120)
    plt.close()


def plot_price_of_safety_dual(
    alphas: List[float],
    net_benefits: List[float],
    recalls: List[float],
    save_path: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(alphas, net_benefits, "o-", color=COLORS[0], label="net benefit")
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("net benefit")
    ax2 = ax1.twinx()
    ax2.plot(alphas, recalls, "s--", color=COLORS[1], label="recall")
    ax2.set_ylabel("recall")
    ax1.set_title("Price of safety")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_latency_cdf(latencies: Dict[str, np.ndarray], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (name, vals) in enumerate(latencies.items()):
        if vals is None or len(vals) == 0:
            continue
        xs = np.sort(np.asarray(vals, dtype=float))
        ys = np.linspace(0, 1, len(xs))
        ax.plot(xs, ys, color=COLORS[i % len(COLORS)], label=name)
    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDFs")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_matched_volume_bars(names: List[str], p99: List[float], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, p99, color=COLORS[0])
    ax.set_ylabel("p99 latency (ms)")
    ax.set_title("Matched-volume p99 latency")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_violation_bars(names: List[str], rates: List[float], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, rates, color=COLORS[5])
    ax.axhline(0.10, color="#f44336", linestyle="--", label="alpha=0.10")
    ax.set_ylabel("violation rate")
    ax.set_title("FDP violation rates")
    plt.xticks(rotation=35, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_tau_histogram(taus: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.asarray(taus, dtype=float), bins=20, color=COLORS[0])
    ax.set_xlabel("tau")
    ax.set_ylabel("count")
    ax.set_title("Break-even tau histogram")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_ltt_validity_bars(
    conditions: List[str],
    violation_rates: List[float],
    save_path: Path,
    title: str = "LTT Validity: Violation Rates",
) -> None:
    """Bar chart: violation rates across i.i.d., autocorrelated, drifted."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS[4], COLORS[3], COLORS[5]]
    x = np.arange(len(conditions))
    bars = ax.bar(x, violation_rates, color=colors[:len(conditions)], alpha=0.8)
    ax.axhline(0.15, color="#f44336", linestyle="--", label="15% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Violation Rate")
    ax.set_title(title)
    ax.legend()
    for bar, val in zip(bars, violation_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
