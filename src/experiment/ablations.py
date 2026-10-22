"""
OFAT ablation driver for ISM.

Sweeps one factor at a time while holding others at defaults.
Factors: alpha_FMR, calibration on/off, threshold static/online,
defer on/off, SSD size, cost multiplier, workload type.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AblationConfig:
    """Configuration for one ablation run."""
    factor: str
    value: Any
    seed: int
    defaults: Dict[str, Any]


@dataclass
class AblationResult:
    """Result of one ablation run."""
    factor: str
    value: Any
    seed: int
    mean_fmr: float
    pooled_fmr: float
    net_benefit: float
    migration_activity: float
    total_migrations: int


def generate_ablation_grid(config: Dict[str, Any]) -> List[AblationConfig]:
    """
    Generate the full OFAT ablation grid from config.

    Returns list of AblationConfig, one per (factor, value, seed) triple.
    """
    exp_cfg = config.get("experiment", {})
    n_seeds = config.get("n_seeds", 10)
    master_seed = config.get("seed", 42)

    defaults = {
        "alpha_fmr": config["calibration"]["alpha_fmr"],
        "calibration": "on",
        "threshold_mode": "online",
        "defer": "on",
        "ssd_fraction": config["budget"]["ssd_fraction"],
        "cost_multiplier": 1.0,
        "workload": "bursty",
    }

    sweeps = {
        "alpha_fmr": exp_cfg.get("alpha_sweep", [0.05, 0.10, 0.20, 0.30]),
        "calibration": ["on", "off"],
        "threshold_mode": ["static", "online"],
        "defer": ["on", "off"],
        "ssd_fraction": exp_cfg.get("ssd_sweep", [0.05, 0.10, 0.20]),
        "cost_multiplier": exp_cfg.get("cost_sweep", [0.5, 1.0, 2.0]),
        "workload": ["steady", "bursty", "changing_hot_set"],
    }

    grid = []
    for factor, values in sweeps.items():
        for val in values:
            for s in range(n_seeds):
                seed = master_seed + s
                grid.append(AblationConfig(
                    factor=factor,
                    value=val,
                    seed=seed,
                    defaults=defaults.copy(),
                ))

    return grid


def format_ablation_table(results: List[AblationResult]) -> str:
    """Format ablation results as a readable table."""
    if not results:
        return "No ablation results."

    lines = [
        f"{'Factor':<20} {'Value':<15} {'Seed':<6} {'MeanFMR':<10} "
        f"{'PooledFMR':<11} {'NetBen':<12} {'Activity':<10} {'Migs':<8}",
        "-" * 92,
    ]
    for r in results:
        lines.append(
            f"{r.factor:<20} {str(r.value):<15} {r.seed:<6} "
            f"{r.mean_fmr:<10.4f} {r.pooled_fmr:<11.4f} "
            f"{r.net_benefit:<12.1f} {r.migration_activity:<10.4f} "
            f"{r.total_migrations:<8}"
        )
    return "\n".join(lines)
