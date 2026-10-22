# Uncertainty-Aware Predictive Data Migration

## Problem
Decide which HDD extents to migrate to SSD before the next horizon so that wasted migrations are risk-controlled at a target false-migration rate, without a silent do-nothing gate.

## Base paper and gap
Learn-then-Test (Angelopoulos et al.) supplies finite-sample testing over a policy grid. The gap is deploying that machinery on a storage migration gate with a physical break-even threshold, two score families, and online/defer variants, rather than unconstrained classification risk.

## Method
Family 1 migrates when a raw lower quantile meets tau*(1+delta). Family 2 migrates when isotonic-calibrated P(Y>=tau|x) meets p*, with logistic trained on the first half of Cal-A and isotonic on the second half, features [q05,q10,q20,q30,q50,q90,spread,tau]. Nested testing walks from conservative to liberal; Bonferroni spends delta_LTT over families actually tested (at most 2). The risk_controlled policy uses the certified family with higher calibration net benefit. Calibration units are pooled Cal-B across training seeds plus cheap calibration-only seeds, sharded by a stable hash of extent id (approximate because shards in an epoch are correlated). The cost model, regime gate, and baseline implementations were not changed.

## Setup
Synthetic noisy_bursty and stress (changing_hot_set drift plus noisy forecasts). Fast replay, five or more seeds. Quantiles {0.05,0.10,0.20,0.30,0.50,0.90}, XGBoost n_estimators=100, subsample=0.5. SimPy capped slices <=50k requests on three seeds. MSRC hm_0 and prn_0 at 16 MiB extents.

## Results
See final_tables.md and targets.md. Numbers below are copied from those tables only.

### Headline (noisy_bursty, alpha=0.10)
- risk_controlled: migrations=19.8, wasted=5.6, FMR=0.2781, FDP=0.0918, activity=0.1611, NB=12976.2, recall=0.4410
- point_forecast: migrations=17.0, wasted=3.4, FMR=0.2009, FDP=0.0703, activity=0.1889, NB=13034.5, recall=0.3627
- static_threshold: migrations=76.6, wasted=62.0, FMR=0.8090, FDP=0.5941, activity=0.6222, NB=2504.6, recall=0.4651

## Limitations
Device latencies, bandwidth, interference, write-back, and SSD wear are assumptions. Synthetic workloads are not production traces. (epoch x shard) units are correlated within an epoch, so the LTT guarantee is approximate. The online tracker is a heuristic. Silent-gate warnings, if any, are limitations of the certified cell, not a reason to retune tau or the workload.

## Reproduction
1. python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
2. .venv\Scripts\python.exe -m pytest tests -q
3. .venv\Scripts\python.exe regime_gate.py
4. make all
5. .venv\Scripts\python.exe demo.py

Each (seed, scenario) job is results/jobs/{scenario}_seed_{seed}.json. Completed jobs are reused; stale caches from prior broken q10 deployments were discarded before the fresh pass.
