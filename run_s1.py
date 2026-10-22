"""S1 runner: generate synthetic workloads, attempt MSRC load, produce plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.data.synthetic import generate_workload
from src.data.msrc_loader import load_msrc_volumes
from src.viz.plots import plot_workload_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "s1_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("S1: Data Loaders & Synthetic Generator")
    print("=" * 60)

    # --- MSRC ---
    print("\n[1/3] MSRC traces")
    msrc_data, _missing = load_msrc_volumes(config, stop_if_missing=False)
    msrc_loaded = [v for v, d in msrc_data.items() if d is not None]
    print(f"  Loaded MSRC volumes: {msrc_loaded or 'none (will use synthetic)'}")

    # --- Synthetic ---
    print("\n[2/3] Synthetic workloads")
    wtypes = config["data"]["synthetic"]["workload_types"]
    syn_summary = []

    for wtype in wtypes:
        seed = abs(config["seed"] + hash(wtype) % 10000) % (2**31)
        df, meta = generate_workload(config, seed=seed, workload_type=wtype)
        syn_summary.append({
            "workload": wtype,
            "seed": seed,
            "n_rows": len(df),
            "n_windows": meta["n_windows"],
            "n_active_extents": int(df["extent"].nunique()),
            "mean_lambda": round(meta["mean_lambda"], 1),
            "n_spikes": meta["n_spikes"],
            "n_decoys": meta["n_decoys"],
            "has_drift": meta["drift"] is not None,
        })
        print(f"  {wtype:20s}  rows={len(df):>7,}  lambda={meta['mean_lambda']:>6.1f}  "
              f"spikes={meta['n_spikes']}  drift={'Y' if meta['drift'] else 'N'}")

        # Plots for first seed of each type
        plot_workload_summary(df, meta, plots_dir)

    # --- Summary table ---
    print("\n[3/3] Summary")
    header = f"{'Workload':<20} {'Rows':>8} {'Windows':>8} {'Extents':>8} {'Lambda':>7} {'Spikes':>7} {'Drift':>6}"
    print(header)
    print("-" * len(header))
    for s in syn_summary:
        print(f"{s['workload']:<20} {s['n_rows']:>8,} {s['n_windows']:>8} "
              f"{s['n_active_extents']:>8} {s['mean_lambda']:>7.1f} "
              f"{s['n_spikes']:>7} {'Y' if s['has_drift'] else 'N':>6}")

    # Save summary
    with open(results_dir / "s1_summary.json", "w") as f:
        json.dump(syn_summary, f, indent=2)
    print(f"\n  Summary saved to {results_dir / 's1_summary.json'}")
    print(f"  Plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
