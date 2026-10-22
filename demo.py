"""Load cached results, print a policy comparison, save key figures. Target: < 1 minute."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from aggregate_final import main as aggregate

ROOT = Path(__file__).parent


def main() -> None:
    aggregate()
    targets = ROOT / "results" / "targets.md"
    tables = ROOT / "results" / "final_tables.md"
    print(targets.read_text(encoding="utf-8"))
    print("--- headline rows from final_tables.md ---")
    for line in tables.read_text(encoding="utf-8").splitlines():
        if line.startswith("| risk_controlled") or line.startswith("| point_forecast") or line.startswith("| static_threshold"):
            print(line)
    figs = sorted((ROOT / "results" / "figures").glob("*.png"))
    print(f"figures={len(figs)}")
    for f in figs:
        print(f.name)


if __name__ == "__main__":
    main()
