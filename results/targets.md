# Targets

- T1: NOT MET — mean test FDP=0.0918 with one-sided 95% bootstrap CI upper limit [0.0721, 0.1115] exceeding alpha=0.10; recall proposed=0.4410 vs 50% of point_forecast 0.3627; share of seed-blocks with FDP>alpha=0.7357.
- T2: NOT VERIFIED — matched-volume comparison was conducted for point_forecast and static_threshold but lacks lru_promote; during burst intervals single-server HDD queue experiences transient saturation (instantaneous arrival rate exceeds service capacity) driving p99 latency to ~114,000–125,000 ms.
- T3: NOT MET — proposed wasted migrations (5.60 in noisy_bursty, 5.60 in stress) exceed point_forecast (3.40 in noisy_bursty, 4.60 in stress) and safety_buffer (2.40 in noisy_bursty, 3.20 in stress); block violation rates (share of seed-blocks with FDP > 0.10) for proposed are 0.7357 (noisy_bursty) and 0.3249 (stress), which are higher than point_forecast (0.4833 and 0.3140) and safety_buffer (0.3840 and 0.2499).
- T4: NOT MET — under stress scenario drift, online threshold tracking does not improve over static: online FMR=0.2115 and net benefit=48342.6 vs static FMR=0.1872 and net benefit=49732.5 (activity 0.3500 vs 0.3444, recall identical at 0.4859).
- T5: MET — mean activity=0.1611 (> 0.05); silent_flags=0; warnings recorded in jobs.

## A6 gate-fix success (noisy_bursty, alpha=0.10, 5+ seeds)
- MET — activity=0.1611 (need >0.05); recall vs 50% of point_forecast: proposed=0.4410 point=0.3627; mean test FDP=0.0918; migrations by alpha=[16.4, 19.8, 25.8, 28.4]; monotone=True.
- Families chosen per alpha: ["alpha=0.05 family=['quantile', 'probability', 'quantile', 'probability', 'probability']", "alpha=0.10 family=['probability', 'probability', 'quantile', 'quantile', 'probability']", "alpha=0.20 family=['quantile', 'probability', 'probability', 'quantile', 'probability']", "alpha=0.30 family=['probability', 'probability', 'quantile', 'quantile', 'probability']"]

Limitation: shard units within an epoch are correlated, so the LTT guarantee is approximate. A silent-gate warning is recorded whenever certified activity < 0.05 or recall < 25% of point_forecast; the policy still runs.

# Headline table (noisy_bursty, alpha=0.10, static, defer off)

| policy | migrations | wasted | pooled FMR | mean FDP | activity | net benefit | recall | frac FDP>alpha |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| risk_controlled | 19.8 | 5.6 | 0.2781 | 0.0918 | 0.1611 | 12976.2 | 0.4410 | 0.7357 |
| point_forecast | 17.0 | 3.4 | 0.2009 | 0.0703 | 0.1889 | 13034.5 | 0.3627 | 0.4833 |
| static_threshold | 76.6 | 62.0 | 0.8090 | 0.5941 | 0.6222 | 2504.6 | 0.4651 | 1.0000 |
| safety_buffer | 15.8 | 2.4 | 0.1490 | 0.0532 | 0.1778 | 13116.8 | 0.3460 | 0.3840 |
| lru_promote | 4427.6 | 3962.8 | 0.8950 | 0.8947 | 1.0000 | -1138551.7 | 0.9289 | 1.0000 |
| hdd_only | 0.0 | 0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0000 | 0.0000 |
| all_ssd | 0.0 | 0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0000 | 0.0000 |
| oracle | 25.4 | 0.0 | 0.0000 | 0.0000 | 0.2167 | 14035.1 | 1.0000 | 0.0000 |
