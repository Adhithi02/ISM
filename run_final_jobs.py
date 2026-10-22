"""Independent fresh fast-replay jobs. One file per (seed, scenario)."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import yaml

from src.experiment.full_job import run_one_job

ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["noisy_bursty", "stress"], required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-cal-only", type=int, default=20)
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--no-ofat", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    if args.reduced:
        config["data"]["synthetic"]["n_windows"] = min(1000, config["data"]["synthetic"]["n_windows"])
        n_cal = min(args.n_cal_only, 5)
    else:
        n_cal = args.n_cal_only

    out_dir = ROOT / "results" / "jobs"
    log_dir = ROOT / "results" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed] if args.seed is not None else list(range(config["seed"], config["seed"] + args.seeds))
    for seed in seeds:
        path = out_dir / f"{args.scenario}_seed_{seed}.json"
        if path.exists():
            print(f"skip finished {path.name}", flush=True)
            continue
        log_path = log_dir / f"{args.scenario}_seed_{seed}.log"
        print(f"start {path.name}", flush=True)
        try:
            payload = run_one_job(
                config, seed, args.scenario,
                n_eval_seeds=args.seeds if args.seed is None else args.seeds,
                n_cal_only=n_cal,
                include_hybrid=args.hybrid,
                reduced=args.reduced,
                ofat=not args.no_ofat,
            )
            path.write_text(json.dumps(payload, indent=2))
            log_path.write_text(f"finished {path.name} reduced={payload.get('reduced')}\n")
            print(path.name, flush=True)
        except Exception:
            log_path.write_text(traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
