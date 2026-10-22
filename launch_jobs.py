"""Launch independent (seed, scenario) jobs with a 15-minute cap and reduced retry."""
from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = ROOT / "results" / "logs"
JOBS = ROOT / "results" / "jobs"
TIMEOUT = 15 * 60
WORKERS = 2


def run_one(cmd, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
        try:
            return proc.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            handle.write("\nTIMEOUT: killed after 15 minutes\n")
            return 124


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    reduced_first = "--reduced" in sys.argv
    JOBS.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    seeds = list(range(42, 42 + n_seeds))
    specs = [(scenario, seed) for scenario in ("noisy_bursty", "stress") for seed in seeds]

    def launch(scenario: str, seed: int, reduced: bool) -> tuple[str, int, int, bool]:
        path = JOBS / f"{scenario}_seed_{seed}.json"
        if path.exists():
            return scenario, seed, 0, True
        tag = "reduced" if reduced else "full"
        log = LOG / f"{scenario}_seed_{seed}_{tag}.log"
        cmd = [str(PY), "-u", "run_final_jobs.py", "--scenario", scenario, "--seed", str(seed), "--seeds", str(n_seeds)]
        if reduced:
            cmd.append("--reduced")
        print(f"start {path.name} {tag}", flush=True)
        rc = run_one(cmd, log)
        print(f"rc {path.name} {tag} {rc} exists={path.exists()}", flush=True)
        return scenario, seed, rc, path.exists()

    first = [(s, sd, reduced_first) for s, sd in specs]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(lambda t: launch(*t), first))

    missing = [(s, sd) for s, sd in specs if not (JOBS / f"{s}_seed_{sd}.json").exists()]
    if missing:
        print("retry reduced", [f"{s}_{sd}" for s, sd in missing], flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(lambda t: launch(t[0], t[1], True), missing))


if __name__ == "__main__":
    main()
