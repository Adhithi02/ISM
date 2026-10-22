"""
S7 Runner: Full Experiments, Ablations, Statistics, Final Figures.

Runs:
1. Multi-seed experiment (10 seeds x bursty workload).
2. OFAT ablation (alpha_FMR sweep).
3. Success target evaluation (T1-T5).
4. Statistical tests (Wilcoxon, bootstrap CIs).
5. Final figures and summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed

from src.experiment.runner import run_single_experiment, RunResult
from src.experiment.metrics import evaluate_guarantee_t1, similar_net_benefit
from src.experiment.stats import paired_wilcoxon, bootstrap_ci
from src.viz.plots import (
    plot_coverage_calibration,
    plot_price_of_safety,
    plot_ltt_validity_bars,
    plot_baseline_bars,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S7 Full Experiments")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--n-seeds", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    n_seeds = args.n_seeds or config.get("n_seeds", 10)
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("S7: Full Experiments, Ablations, Statistics, Final Figures")
    print("=" * 70)

    master_seed = config["seed"]
    alpha_fmr = config["calibration"]["alpha_fmr"]

    # 1. Multi-seed experiment
    print(f"\n[1/5] Running {n_seeds}-seed experiment (bursty) ...")
    seeds = [master_seed + s for s in range(n_seeds)]

    results: list[RunResult] = Parallel(n_jobs=2, verbose=5)(
        delayed(run_single_experiment)(config, seed, "bursty")
        for seed in seeds
    )

    print(f"\n  Completed {len(results)} runs")
    mean_fmrs = np.array([r.decision_metrics.mean_fmr for r in results])
    net_benefits = np.array([r.decision_metrics.net_benefit for r in results])
    certified = [r.ltt_certified for r in results]

    print(f"  Mean FMR across seeds: {np.mean(mean_fmrs):.4f} +/- {np.std(mean_fmrs):.4f}")
    print(f"  Mean net benefit: {np.mean(net_benefits):.1f} +/- {np.std(net_benefits):.1f}")
    print(f"  LTT certified: {sum(certified)}/{len(certified)}")

    # 2. Alpha sweep ablation
    print(f"\n[2/5] Alpha_FMR sweep ablation ...")
    alpha_sweep = config["experiment"]["alpha_sweep"]
    ablation_results = {}

    for alpha in alpha_sweep:
        print(f"  alpha_FMR = {alpha} ...", end="", flush=True)
        ab_results = Parallel(n_jobs=2, verbose=0)(
            delayed(run_single_experiment)(config, seed, "bursty", alpha_fmr=alpha)
            for seed in seeds[:min(5, n_seeds)]
        )
        mean_nb = float(np.mean([r.decision_metrics.net_benefit for r in ab_results]))
        mean_fmr = float(np.mean([r.decision_metrics.mean_fmr for r in ab_results]))
        ablation_results[alpha] = {"net_benefit": mean_nb, "mean_fmr": mean_fmr}
        print(f" FMR={mean_fmr:.4f}, NB={mean_nb:.1f}")

    # Price of safety plot
    plot_price_of_safety(
        alphas=list(ablation_results.keys()),
        net_benefits=[v["net_benefit"] for v in ablation_results.values()],
        save_path=plots_dir / "price_of_safety.png",
    )
    print(f"  Saved: plots/price_of_safety.png")

    # 3. Success targets
    print(f"\n[3/5] Evaluating success targets ...")

    # T1: Mean FMR <= alpha in >= 90% of blocks
    t1 = evaluate_guarantee_t1(mean_fmrs, alpha_fmr)
    t1_status = "MET" if t1["met"] else "NOT MET"
    print(f"  T1: {t1_status} - {t1['criterion']} "
          f"(actual: {t1['pass_fraction']:.1%})")

    # T3: >= 30% fewer wasted migrations vs point forecast
    # Run point forecast baseline for comparison
    pf_results = Parallel(n_jobs=2, verbose=0)(
        delayed(run_single_experiment)(
            config, seed, "bursty", policy="point_forecast"
        )
        for seed in seeds[:min(5, n_seeds)]
    )
    method_wasted = np.array([r.decision_metrics.wasted_migrations for r in results[:min(5, n_seeds)]])
    pf_wasted = np.array([r.decision_metrics.wasted_migrations for r in pf_results])
    if len(pf_wasted) > 0 and np.mean(pf_wasted) > 0:
        reduction = 1.0 - np.mean(method_wasted) / np.mean(pf_wasted)
        t3_met = reduction >= 0.30
    else:
        reduction = 0.0
        t3_met = False
    t3_status = "MET" if t3_met else "NOT MET"
    print(f"  T3: {t3_status} - Wasted reduction vs baseline: {reduction:.1%}")

    # T5: Gate never silently 0 migrations
    activities = [r.decision_metrics.migration_activity for r in results]
    t5_met = all(a > 0 for a in activities)
    t5_status = "MET" if t5_met else "NOT MET"
    print(f"  T5: {t5_status} - All runs have migration activity > 0")

    # 4. Statistical tests
    print(f"\n[4/5] Statistical tests ...")

    # Bootstrap CI for mean FMR
    fmr_ci = bootstrap_ci(mean_fmrs, n_bootstrap=config["experiment"]["n_bootstrap"])
    print(f"  Mean FMR 95% CI: [{fmr_ci.ci_lower:.4f}, {fmr_ci.ci_upper:.4f}]")

    # Bootstrap CI for net benefit
    nb_ci = bootstrap_ci(net_benefits, n_bootstrap=config["experiment"]["n_bootstrap"])
    print(f"  Net Benefit 95% CI: [{nb_ci.ci_lower:.1f}, {nb_ci.ci_upper:.1f}]")

    # Wilcoxon test requires the full paired baseline sample.
    wilcoxon_result = None
    if len(method_wasted) >= 10 and len(pf_wasted) >= 10:
        wt = paired_wilcoxon(
            method_wasted.astype(float), pf_wasted.astype(float)
        )
        wilcoxon_result = wt.__dict__
        print(f"  Wilcoxon (wasted mig): p={wt.p_value:.6f}, significant={wt.significant}")
    else:
        print("  Wilcoxon (wasted mig): NOT VERIFIED (fewer than 10 paired seeds)")

    # 5. LTT validity plot. The multi-seed runner does not estimate
    # exchangeability violation rates, so do not plot fabricated values.
    print(f"\n[5/5] Generating final figures ...")
    plot_ltt_validity_bars(
        conditions=["i.i.d.", "AR(1)", "Drifted"],
        violation_rates=[float("nan")] * 3,
        save_path=plots_dir / "ltt_validity_bars.png",
    )
    print(f"  Saved: plots/ltt_validity_bars.png")

    # Coverage calibration plot from the stage-4 evaluation, if available.
    quantile_levels = config["models"]["quantile_levels"]
    s4_path = results_dir / "s4_summary.json"
    if s4_path.exists():
        with open(s4_path) as f:
            s4_summary = json.load(f)
        eval_table = s4_summary.get("eval_table", [])
        coverage = {
            float(row["quantile"]): float(row["coverage_below"])
            for row in eval_table
            if "quantile" in row and "coverage_below" in row
        }
        if all(q in coverage for q in quantile_levels):
            plot_coverage_calibration(
                quantile_levels,
                [coverage[q] for q in quantile_levels],
                save_path=plots_dir / "coverage_calibration.png",
            )
            print(f"  Saved: plots/coverage_calibration.png")
        else:
            print("  Coverage calibration: NOT VERIFIED (incomplete S4 table)")
    else:
        print("  Coverage calibration: NOT VERIFIED (S4 summary missing)")

    # Save all results
    summary = {
        "multi_seed": {
            "n_seeds": n_seeds,
            "mean_fmr": round(float(np.mean(mean_fmrs)), 4),
            "std_fmr": round(float(np.std(mean_fmrs)), 4),
            "mean_net_benefit": round(float(np.mean(net_benefits)), 1),
            "std_net_benefit": round(float(np.std(net_benefits)), 1),
            "ltt_certified": sum(certified),
            "fmr_ci": {"lower": fmr_ci.ci_lower, "upper": fmr_ci.ci_upper},
            "nb_ci": {"lower": nb_ci.ci_lower, "upper": nb_ci.ci_upper},
        },
        "alpha_sweep": {str(k): v for k, v in ablation_results.items()},
        "success_targets": {
            "T1": {"status": t1_status, **t1},
            "T2": {
                "status": "NOT VERIFIED",
                "reason": "Matched-volume latency comparison was not run by S7.",
            },
            "T3": {
                "status": t3_status,
                "wasted_reduction": round(reduction, 4),
                "baseline_policy": "point_forecast",
            },
            "T4": {
                "status": "NOT VERIFIED",
                "reason": "Post-drift static-versus-online target was not evaluated in S7.",
            },
            "T5": {"status": t5_status, "all_active": t5_met},
        },
        "statistics": {"wilcoxon_wasted_migrations": wilcoxon_result},
    }

    out_file = results_dir / "s7_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nS7 Summary written to {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
