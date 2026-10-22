"""Final fresh pass; does not read or invoke S7 or prior result caches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiment.runner import run_single_experiment


ROOT = Path(__file__).parent


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    # The full-strength five-seed pass exceeded the 20-minute execution budget.
    # Keep the model configuration unchanged and reduce only the workload slice.
    config["data"]["synthetic"]["n_windows"] = 500
    seeds = [config["seed"]]
    rows = []
    for seed in seeds:
        for alpha in config["experiment"]["alpha_sweep"]:
            result = run_single_experiment(
                config, seed, "noisy_bursty", alpha_fmr=alpha,
                policy="risk_controlled",
            )
            rows.append({
                "seed": seed, "alpha": alpha, "policy": "risk_controlled",
                **result.decision_metrics.__dict__,
                "ltt_certified": result.ltt_certified,
                "q": result.best_q, "delta": result.best_delta,
            })
        result = run_single_experiment(config, seed, "noisy_bursty", policy="point_forecast")
        rows.append({
            "seed": seed, "alpha": None, "policy": "point_forecast",
            **result.decision_metrics.__dict__,
            "ltt_certified": False, "q": result.best_q, "delta": result.best_delta,
        })
    summary = {
        "regime": "noisy_bursty",
        "seeds": seeds,
        "rows": rows,
        "status": "fresh_reduced_slice_full_strength_model_run",
        "reduced": True,
    }
    (ROOT / "results" / "final_fresh.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Final Targets",
        "",
        "- T1: NOT VERIFIED — reduced one-seed slice; block-level guarantee aggregation is underpowered.",
        "- T2: NOT VERIFIED — SimPy policy-specific matched-volume run is not included in this bounded pass.",
        "- T3: NOT VERIFIED — reduced slice lacks full baseline and recall comparison.",
        "- T4: NOT VERIFIED — stress drift activity/net-benefit/recall table is not run.",
        "- T5: NOT VERIFIED — migration activity is reported, but one reduced slice is insufficient.",
        "",
        "# Headline table",
        "",
        "| alpha | policy | migrations | wasted | pooled FMR | mean FDP | activity | net benefit |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for alpha in config["experiment"]["alpha_sweep"]:
        subset = [r for r in rows if r["policy"] == "risk_controlled" and r["alpha"] == alpha]
        lines.append(
            f"| {alpha:.2f} | risk_controlled | "
            f"{np.mean([r['total_migrations'] for r in subset]):.1f} | "
            f"{np.mean([r['wasted_migrations'] for r in subset]):.1f} | "
            f"{np.mean([r['pooled_fmr'] for r in subset]):.4f} | "
            f"{np.mean([r['mean_fmr'] for r in subset]):.4f} | "
            f"{np.mean([r['migration_activity'] for r in subset]):.4f} | "
            f"{np.mean([r['net_benefit'] for r in subset]):.1f} |"
        )
    subset = [r for r in rows if r["policy"] == "point_forecast"]
    lines.append(
        f"| — | point_forecast | {np.mean([r['total_migrations'] for r in subset]):.1f} | "
        f"{np.mean([r['wasted_migrations'] for r in subset]):.1f} | "
        f"{np.mean([r['pooled_fmr'] for r in subset]):.4f} | "
        f"{np.mean([r['mean_fmr'] for r in subset]):.4f} | "
        f"{np.mean([r['migration_activity'] for r in subset]):.4f} | "
        f"{np.mean([r['net_benefit'] for r in subset]):.1f} |"
    )
    (ROOT / "results" / "targets.md").write_text("\n".join(lines) + "\n")
    (ROOT / "results" / "final_tables.md").write_text("\n".join(lines) + "\n")
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    alpha_rows = []
    for alpha in config["experiment"]["alpha_sweep"]:
        alpha_rows.append(np.mean([r["net_benefit"] for r in rows if r["policy"] == "risk_controlled" and r["alpha"] == alpha]))
    plt.figure(figsize=(5, 3))
    plt.plot(config["experiment"]["alpha_sweep"], alpha_rows, marker="o")
    plt.xlabel("alpha")
    plt.ylabel("net benefit")
    plt.tight_layout()
    plt.savefig(ROOT / "results" / "figures" / "price_of_safety.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
