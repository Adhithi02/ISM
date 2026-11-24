# Targets

- T1: NOT MET — mean=0.0918 95% CI [0.0721, 0.1115]; recall proposed=0.4410 vs 50% of point_forecast 0.3627; share of seed-blocks with FDP>alpha=0.7357.
- T2: MET — hdd_only hits=[0.0, 0.0, 0.0] all_ssd hits=[1.0, 1.0, 1.0] p99 triples=[(117057.93308758001, 0.4567567976666041, 114795.8935093259), (122878.19567464248, 0.45721112405434733, 118286.367861138), (125230.6435818872, 0.4572394184471344, 120438.23354156222)].
- T3: MET — wasted proposed=5.60, point=3.40, static=62.00; violation proposed frac>alpha=0.7357, point=nan, static=nan. Stress proposed wasted=5.60.
- T4: MET — static FMR=0.1872 act=0.3444 nb=49732.5 rec=0.4859; online FMR=0.2115 act=0.3500 nb=48342.6 rec=0.4859.
- T5: MET — mean activity=0.1611; silent_flags=0; warnings recorded in jobs.

## A6 gate-fix success (noisy_bursty, alpha=0.10, 5+ seeds)
- MET — activity=0.1611 (need >0.05); recall vs 50% of point_forecast: proposed=0.4410 point=0.3627; mean test FDP=0.0918; migrations by alpha=[16.4, 19.8, 25.8, 28.4]; monotone=True.
- Families chosen per alpha: ["alpha=0.05 family=['quantile', 'probability', 'quantile', 'probability', 'probability']", "alpha=0.10 family=['probability', 'probability', 'quantile', 'quantile', 'probability']", "alpha=0.20 family=['quantile', 'probability', 'probability', 'quantile', 'probability']", "alpha=0.30 family=['probability', 'probability', 'quantile', 'quantile', 'probability']"]

Limitation: shard units within an epoch are correlated, so the LTT guarantee is approximate. A silent-gate warning is recorded whenever certified activity < 0.05 or recall < 25% of point_forecast; the policy still runs.

# Headline table (noisy_bursty, alpha=0.10, static, defer off)

| policy | migrations | wasted | pooled FMR | mean FDP | activity | net benefit | recall | frac FDP>alpha |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| risk_controlled | 19.8 | 5.6 | 0.2781 | 0.0918 | 0.1611 | 12976.2 | 0.4410 | 0.7357 |
| point_forecast | 17.0 | 3.4 | 0.2009 | 0.0703 | 0.1889 | 13034.5 | 0.3627 | nan |
| static_threshold | 76.6 | 62.0 | 0.8090 | 0.5941 | 0.6222 | 2504.6 | 0.4651 | nan |
| safety_buffer | 15.8 | 2.4 | 0.1490 | 0.0532 | 0.1778 | 13116.8 | 0.3460 | nan |
| lru_promote | 4427.6 | 3962.8 | 0.8950 | 0.8947 | 1.0000 | -1138551.7 | 0.9289 | nan |
| hdd_only | 0.0 | 0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0000 | nan |
| all_ssd | 0.0 | 0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0000 | nan |
| oracle | 25.4 | 0.0 | 0.0000 | 0.0000 | 0.2167 | 14035.1 | 1.0000 | nan |
