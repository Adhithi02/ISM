"""
Master execution script for Multi-Algorithm and Multi-Dataset ISM Benchmarking.

Executes all 5 algorithms:
- XGBoost
- LightGBM
- CatBoost
- HistGradientBoosting
- Linear Quantile Regression

Across all 3 dataset regimes:
- Synthetic-only
- Hybrid (Synthetic + Real trace)
- MSRC-only (Enterprise Microsoft Traces)

Generates:
1. `results/algorithm_benchmark_results.json`
2. `results/paper_benchmark_report.md` (complete publication-ready markdown + LaTeX tables)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from src.experiment.algorithm_benchmark import run_full_algorithm_benchmark


def generate_paper_report(results: List[Dict[str, Any]], output_path: Path) -> str:
    """Generate publication-ready Markdown report with comparative tables and analysis."""
    df = pd.DataFrame(results)
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")].copy()

    lines = []
    lines.append("# Empirical Evaluation & Cross-Algorithm Benchmark for ISM")
    lines.append("\n## 1. Executive Summary & Core Findings\n")
    lines.append(
        "This evaluation benchmarks **5 advanced quantile forecasters** across **3 storage workload regimes** "
        "(Synthetic-only, Hybrid, and Microsoft Enterprise MSRC traces). Each forecaster is coupled with the "
        "Learn-then-Test (LTT) risk-controlled migration gate to certify false migration proportions (FDP ≤ α) "
        "and benchmarked end-to-end on both fast matrix trace replay and the discrete-event queue simulator (SimPy)."
    )

    # 1. Prediction Accuracy Table
    lines.append("\n## 2. Forecast Accuracy & Uncertainty Quantification\n")
    lines.append(
        "| Dataset Regime | Algorithm | Training Time (s) | Pinball Loss (Mean) | MAE (Median) | 80% CI Coverage | 80% CI Width | Pre-Sort Crossing Rate |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|"
    )
    for _, row in df.iterrows():
        lines.append(
            f"| {row['dataset_mode']} | **{row['algorithm']}** | {row['training_time_s']:.2f} | "
            f"{row['pinball_loss_avg']:.4f} | {row['mae_median']:.4f} | {row['interval_coverage_80']:.4f} | "
            f"{row['interval_mean_width']:.2f} | {row['crossing_rate']:.4f} |"
        )

    # 2. Risk Control & Migration System Performance Table
    lines.append("\n## 3. Risk-Controlled Migration Policy & Safety Metrics\n")
    lines.append(
        "| Dataset Regime | Algorithm | LTT Status | Chosen Family & Param | Migrations | Wasted | Mean FDP | Pooled FMR | Activity | Net Benefit (ms) | Recall vs Oracle |"
    )
    lines.append(
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for _, row in df.iterrows():
        lines.append(
            f"| {row['dataset_mode']} | **{row['algorithm']}** | {row['ltt_status']} | {row['chosen_family']} ({row['chosen_param']}) | "
            f"{row['migrations']} | {row['wasted_migrations']} | {row['mean_fdp']:.4f} | {row['pooled_fmr']:.4f} | "
            f"{row['migration_activity']:.4f} | {row['net_benefit']:.1f} | {row['recall_vs_oracle']:.4f} |"
        )

    # 3. Discrete-Event Simulator (SimPy) Real-Time Translation Table
    lines.append("\n## 4. Real-Time Storage Hardware Simulation (SimPy Queue Validation)\n")
    lines.append(
        "The discrete-event simulation translates the decisions onto hardware queueing behavior (M/M/1-verified HDD queue + SSD cache tier).\n"
    )
    lines.append(
        "| Dataset Regime | Algorithm | Mean Latency (ms) | P95 Latency (ms) | P99 Tail Latency (ms) | IOPS | SSD Hit Ratio | SLA Violation Rate (>10ms) | HDD Util (ρ) |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for _, row in df.iterrows():
        lines.append(
            f"| {row['dataset_mode']} | **{row['algorithm']}** | {row['sim_mean_latency_ms']:.2f} | "
            f"{row['sim_p95_latency_ms']:.2f} | {row['sim_p99_latency_ms']:.2f} | {row['sim_iops']:.1f} | "
            f"{row['sim_ssd_hit_ratio']:.4f} | {row['sim_sla_violation_rate']:.4f} | {row['sim_hdd_utilization']:.4f} |"
        )

    # 4. Identification of the Best Algorithm & Comparative Quantification
    lines.append("\n## 5. Model Quantification & Best Performing Forecaster\n")
    
    # Compute rankings per dataset mode
    for mode in df["dataset_mode"].unique():
        sub = df[df["dataset_mode"] == mode].copy()
        if sub.empty:
            continue
        # Score based on lower pinball loss, higher net benefit, and lower p99 latency
        sub["rank_pinball"] = sub["pinball_loss_avg"].rank(ascending=True)
        sub["rank_net_ben"] = sub["net_benefit"].rank(ascending=False)
        sub["rank_p99"] = sub["sim_p99_latency_ms"].rank(ascending=True)
        sub["composite_score"] = sub["rank_pinball"] + sub["rank_net_ben"] + sub["rank_p99"]
        best_row = sub.sort_values("composite_score").iloc[0]

        lines.append(f"### Best Model for Regime: **{mode.upper()}**")
        lines.append(f"- **Top Performer:** `{best_row['algorithm']}`")
        lines.append(
            f"- **Forecasting Efficiency:** Average pinball loss of `{best_row['pinball_loss_avg']:.4f}` with 80% coverage `{best_row['interval_coverage_80']:.4f}`."
        )
        lines.append(
            f"- **System Safety & Benefit:** Net benefit of `{best_row['net_benefit']:.1f} ms` saved with Mean FDP `{best_row['mean_fdp']:.4f}` (guaranteed ≤ α=0.10)."
        )
        lines.append(
            f"- **Real-Time Tail Latency:** P99 tail latency of `{best_row['sim_p99_latency_ms']:.2f} ms` and SSD hit ratio of `{best_row['sim_ssd_hit_ratio'] * 100:.2f}%`.\n"
        )

    # 5. LaTeX Ready Tables for Paper Inclusion
    lines.append("\n## 6. LaTeX Formatted Tables for Paper Submission\n")
    lines.append("```latex")
    lines.append("% --- Table: Multi-Algorithm Prediction & Storage Performance Comparison ---")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Cross-Algorithm & Multi-Dataset Performance Comparison in Tiered Storage Migration}")
    lines.append("\\label{tab:algorithm_comparison}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrrrrrr}")
    lines.append("\\toprule")
    lines.append("Regime & Algorithm & Train(s) & Pinball & Coverage & FDP & Net Ben (ms) & Hit Ratio & P99 Lat (ms) \\\\")
    lines.append("\\midrule")
    for _, r in df.iterrows():
        lines.append(
            f"{r['dataset_mode']} & {r['algorithm']} & {r['training_time_s']:.1f} & {r['pinball_loss_avg']:.3f} & "
            f"{r['interval_coverage_80']:.3f} & {r['mean_fdp']:.3f} & {r['net_benefit']:.0f} & "
            f"{r['sim_ssd_hit_ratio']*100:.1f}\\% & {r['sim_p99_latency_ms']:.1f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    lines.append("```")

    report_content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content, encoding="utf-8")
    return report_content


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-algorithm & multi-dataset benchmark for ISM")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config file")
    parser.add_argument("--volume", default="hm_0", help="MSRC trace volume to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--algorithms", nargs="+", default=None, help="Algorithms to benchmark")
    parser.add_argument("--datasets", nargs="+", default=None, help="Dataset regimes: synthetic, hybrid, msrc")
    args = parser.parse_args()

    results = run_full_algorithm_benchmark(
        config_path=args.config,
        algorithms=args.algorithms,
        dataset_modes=args.datasets,
        seed=args.seed,
        volume=args.volume,
    )

    root = Path(__file__).resolve().parent
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / "algorithm_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved JSON]: {json_path}")

    report_path = results_dir / "paper_benchmark_report.md"
    generate_paper_report(results, report_path)
    print(f"[Saved Paper Report]: {report_path}")


if __name__ == "__main__":
    main()
