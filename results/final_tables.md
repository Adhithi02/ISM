# Final tables

Fast-replay jobs: quantiles {0.05,0.10,0.20,0.30,0.50,0.90}, XGBoost n_estimators=100, subsample=0.5, n_jobs=2.
Calibration units are (epoch x extent-shard) with 8 shards from a stable hash of extent id; shards within an epoch are correlated, so the guarantee is approximate.
delta_LTT=0.10 is Bonferroni-split over the families actually tested (at most 2).
Stale S7 caches were not used. Tables are labelled reduced because at least one job used a reduced load.

## noisy_bursty

| Policy | Alpha | Mode | Defer | Family | Migrations | Wasted | Pooled FMR | Mean FDP | Activity | Net benefit | Recall | Frac blocks FDP>alpha |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| risk_controlled | 0.05 | static | False | quantile | 16.4 | 3.8 | 0.2319 | 0.0742 | 0.1556 | 13070.2 | 0.3787 | 0.6576 |
| risk_controlled | 0.05 | static | True | quantile | 16.4 | 3.8 | 0.2319 | 0.0742 | 0.1556 | 13070.2 | 0.3787 | 0.6576 |
| risk_controlled | 0.05 | online | False | quantile | 16.4 | 3.4 | 0.2088 | 0.0658 | 0.1556 | 13016.4 | 0.3787 | 0.5971 |
| risk_controlled | 0.1 | static | False | probability | 19.8 | 5.6 | 0.2781 | 0.0918 | 0.1611 | 12976.2 | 0.4410 | 0.7357 |
| risk_controlled | 0.1 | static | True | probability | 19.8 | 5.6 | 0.2781 | 0.0918 | 0.1611 | 12976.2 | 0.4410 | 0.7357 |
| risk_controlled | 0.1 | online | False | probability | 21.0 | 6.2 | 0.2841 | 0.1057 | 0.1889 | 12835.4 | 0.4410 | 0.6762 |
| risk_controlled | 0.2 | static | False | quantile | 25.8 | 10.6 | 0.4103 | 0.1305 | 0.1833 | 12441.3 | 0.5074 | 0.8429 |
| risk_controlled | 0.2 | static | True | quantile | 25.8 | 10.6 | 0.4103 | 0.1305 | 0.1833 | 12441.3 | 0.5074 | 0.8429 |
| risk_controlled | 0.2 | online | False | quantile | 27.4 | 12.0 | 0.4377 | 0.1796 | 0.2500 | 12316.8 | 0.5157 | 0.8083 |
| risk_controlled | 0.3 | static | False | probability | 28.4 | 13.4 | 0.4705 | 0.1583 | 0.2056 | 12141.1 | 0.5616 | 0.8378 |
| risk_controlled | 0.3 | static | True | probability | 28.4 | 13.4 | 0.4705 | 0.1583 | 0.2056 | 12141.1 | 0.5616 | 0.8378 |
| risk_controlled | 0.3 | online | False | probability | 31.8 | 17.0 | 0.4978 | 0.1847 | 0.2278 | 11404.0 | 0.5147 | 0.7952 |
| point_forecast | — | static | False | — | 17.0 | 3.4 | 0.2009 | 0.0703 | 0.1889 | 13034.5 | 0.3627 | nan |
| safety_buffer | — | static | False | — | 15.8 | 2.4 | 0.1490 | 0.0532 | 0.1778 | 13116.8 | 0.3460 | nan |
| static_threshold | — | static | False | — | 76.6 | 62.0 | 0.8090 | 0.5941 | 0.6222 | 2504.6 | 0.4651 | nan |
| lru_promote | — | static | False | — | 4427.6 | 3962.8 | 0.8950 | 0.8947 | 1.0000 | -1138551.7 | 0.9289 | nan |
| hdd_only | — | static | False | — | 0.0 | 0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0000 | nan |
| all_ssd | — | static | False | — | 0.0 | 0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0000 | nan |
| oracle | — | static | False | — | 25.4 | 0.0 | 0.0000 | 0.0000 | 0.2167 | 14035.1 | 1.0000 | nan |
| proposed_lru_evict | — | static | False | — | 0.4 | 0.0 | 0.0000 | 0.0000 | 0.0111 | 2544.3 | 0.0000 | nan |
| point_forecast_value_evict | — | static | False | — | 17.0 | 3.4 | 0.2009 | 0.0703 | 0.1889 | 13034.5 | 0.3627 | nan |

### Certification per alpha

| Alpha | Family chosen | q | delta | p* | delta_LTT | p_threshold | Units before shards | Units after shards | Migrating units | Smallest p | Certified cells | Cal net benefit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | quantile | 0.500 | 0.000 | 0.700 | 0.1 | 0.0500 | 60.0 | 3600.0 | 3574.6 | 0.000000 | 181.0 | 7994377.7 |
| 0.10 | probability | 0.500 | -0.060 | 0.483 | 0.1 | 0.0500 | 60.0 | 3600.0 | 3591.8 | 0.000000 | 190.8 | 8097449.6 |
| 0.20 | quantile | 0.500 | -0.160 | 0.300 | 0.1 | 0.0500 | 60.0 | 3600.0 | 3599.8 | 0.000000 | 199.0 | 8150542.1 |
| 0.30 | probability | 0.300 | -0.240 | 0.200 | 0.1 | 0.0500 | 60.0 | 3600.0 | 3600.0 | 0.000000 | 205.4 | 8123496.5 |

## stress

| Policy | Alpha | Mode | Defer | Family | Migrations | Wasted | Pooled FMR | Mean FDP | Activity | Net benefit | Recall | Frac blocks FDP>alpha |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| risk_controlled | 0.05 | static | False | quantile | 27.8 | 4.4 | 0.1579 | 0.0760 | 0.3389 | 48516.7 | 0.4338 | 0.3107 |
| risk_controlled | 0.05 | static | True | quantile | 27.8 | 4.4 | 0.1579 | 0.0760 | 0.3389 | 48516.7 | 0.4338 | 0.3107 |
| risk_controlled | 0.05 | online | False | quantile | 27.0 | 3.8 | 0.1420 | 0.0594 | 0.3278 | 48577.4 | 0.4338 | 0.2559 |
| risk_controlled | 0.1 | static | False | probability | 29.8 | 5.6 | 0.1872 | 0.0838 | 0.3444 | 49732.5 | 0.4859 | 0.3249 |
| risk_controlled | 0.1 | static | True | probability | 29.8 | 5.6 | 0.1872 | 0.0838 | 0.3444 | 49732.5 | 0.4859 | 0.3249 |
| risk_controlled | 0.1 | online | False | probability | 30.2 | 6.4 | 0.2115 | 0.0968 | 0.3500 | 48342.6 | 0.4859 | 0.3498 |
| risk_controlled | 0.2 | static | False | probability | 33.4 | 9.0 | 0.2698 | 0.1170 | 0.3722 | 49356.0 | 0.5064 | 0.3445 |
| risk_controlled | 0.2 | static | True | probability | 33.4 | 9.0 | 0.2698 | 0.1170 | 0.3722 | 49356.0 | 0.5064 | 0.3445 |
| risk_controlled | 0.2 | online | False | probability | 37.6 | 13.0 | 0.3350 | 0.1558 | 0.3944 | 49166.6 | 0.5064 | 0.4573 |
| risk_controlled | 0.3 | static | False | probability | 34.8 | 10.4 | 0.2976 | 0.1243 | 0.3722 | 49237.5 | 0.5064 | 0.3732 |
| risk_controlled | 0.3 | static | True | probability | 34.8 | 10.4 | 0.2976 | 0.1243 | 0.3722 | 49237.5 | 0.5064 | 0.3732 |
| risk_controlled | 0.3 | online | False | probability | 38.4 | 14.0 | 0.3535 | 0.1419 | 0.3722 | 48412.3 | 0.5129 | 0.4327 |
| point_forecast | — | static | False | — | 27.0 | 4.6 | 0.1704 | 0.0820 | 0.3278 | 44063.3 | 0.4266 | nan |
| safety_buffer | — | static | False | — | 26.0 | 3.8 | 0.1479 | 0.0678 | 0.3222 | 44079.5 | 0.4136 | nan |
| static_threshold | — | static | False | — | 66.2 | 41.8 | 0.6295 | 0.4696 | 0.6389 | 39695.0 | 0.5336 | nan |
| lru_promote | — | static | False | — | 4898.8 | 4397.0 | 0.8976 | 0.8973 | 1.0000 | -1237567.2 | 0.7446 | nan |
| hdd_only | — | static | False | — | 0.0 | 0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0000 | nan |
| all_ssd | — | static | False | — | 0.0 | 0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0.0000 | nan |
| oracle | — | static | False | — | 30.6 | 0.0 | 0.0000 | 0.0000 | 0.3389 | 32884.3 | 1.0000 | nan |
| proposed_lru_evict | — | static | False | — | 3.4 | 0.0 | 0.0000 | 0.0000 | 0.0722 | 9461.9 | 0.0530 | nan |
| point_forecast_value_evict | — | static | False | — | 27.0 | 4.6 | 0.1704 | 0.0820 | 0.3278 | 44063.3 | 0.4266 | nan |

### Certification per alpha

| Alpha | Family chosen | q | delta | p* | delta_LTT | p_threshold | Units before shards | Units after shards | Migrating units | Smallest p | Certified cells | Cal net benefit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | quantile | 0.500 | 0.000 | 0.617 | 0.1 | 0.0500 | 60.0 | 3600.0 | 3599.8 | 0.000000 | 181.6 | 9340770.6 |
| 0.10 | probability | 0.500 | -0.020 | 0.375 | 0.1 | 0.0500 | 60.0 | 3600.0 | 3600.0 | 0.000000 | 190.6 | 9397420.5 |
| 0.20 | probability | 0.500 | -0.060 | 0.112 | 0.1 | 0.0500 | 60.0 | 3600.0 | 3600.0 | 0.000000 | 200.8 | 9390858.5 |
| 0.30 | probability | 0.500 | -0.080 | 0.075 | 0.1 | 0.0500 | 60.0 | 3600.0 | 3600.0 | 0.000000 | 204.6 | 9365254.6 |

## OFAT ablations (alpha=0.10, noisy_bursty)

| Factor | Value | Migrations | Wasted | Mean FDP | Activity | Net benefit | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| calibration | on | 19.8 | 5.6 | 0.0918 | 0.1611 | 12976.2 | 0.4410 |
| calibration | off | 17.0 | 3.4 | 0.0703 | 0.1889 | 13034.5 | 0.3627 |
| threshold_mode | static | 19.8 | 5.6 | 0.0918 | 0.1611 | 12976.2 | 0.4410 |
| threshold_mode | online | 21.0 | 6.2 | 0.1057 | 0.1889 | 12835.4 | 0.4410 |
| defer | on | 19.8 | 5.6 | 0.0918 | 0.1611 | 12976.2 | 0.4410 |
| defer | off | 19.8 | 5.6 | 0.0918 | 0.1611 | 12976.2 | 0.4410 |
| ssd_fraction | 0.05 | 19.8 | 5.6 | 0.0918 | 0.1611 | 12976.2 | 0.4410 |
| ssd_fraction | 0.1 | 19.8 | 5.6 | 0.0918 | 0.1611 | 12976.2 | 0.4410 |
| ssd_fraction | 0.2 | 19.8 | 5.6 | 0.0918 | 0.1611 | 12976.2 | 0.4410 |
| cost_multiplier | 0.5 | 25.2 | 4.6 | 0.0672 | 0.1834 | 15028.6 | 0.4885 |
| cost_multiplier | 1.0 | 19.8 | 5.6 | 0.0918 | 0.1611 | 12976.2 | 0.4410 |
| cost_multiplier | 2.0 | 17.4 | 9.0 | 0.1115 | 0.1445 | 8943.9 | 0.3846 |

## Stress post-drift time windows (test split is after onset_window=1000 when n_windows=2500)

| Policy | Window | Mean FMR | Activity | Net benefit | Recall |
|---|---|---:|---:|---:|---:|
| risk_controlled static | post-drift W1 seed 42 (test [818, 993], onset 1000) | 0.3015 | 0.6667 | 3017.8 | 0.4411764705882353 |
| risk_controlled static | post-drift W2 seed 42 (test [818, 993], onset 1000) | 0.0000 | 0.2500 | 1185.3 | 0.4411764705882353 |
| risk_controlled static | post-drift W3 seed 42 (test [818, 993], onset 1000) | 0.0000 | 0.0833 | 648.6 | 0.4411764705882353 |
| risk_controlled online | post-drift W1 seed 42 (test [818, 993], onset 1000) | 0.2598 | 0.6667 | 3028.3 | 0.4411764705882353 |
| risk_controlled online | post-drift W2 seed 42 (test [818, 993], onset 1000) | 0.0833 | 0.2500 | 634.9 | 0.4411764705882353 |
| risk_controlled online | post-drift W3 seed 42 (test [818, 993], onset 1000) | 0.0833 | 0.1667 | 643.4 | 0.4411764705882353 |
| point_forecast static | post-drift W1 seed 42 (test [818, 993], onset 1000) | 0.2000 | 0.6667 | 3036.0 | 0.38235294117647056 |
| point_forecast static | post-drift W2 seed 42 (test [818, 993], onset 1000) | 0.1667 | 0.3333 | 627.1 | 0.38235294117647056 |
| point_forecast static | post-drift W3 seed 42 (test [818, 993], onset 1000) | 0.0833 | 0.1667 | 647.3 | 0.38235294117647056 |
| static_threshold static | post-drift W1 seed 42 (test [818, 993], onset 1000) | 0.6900 | 0.9167 | 2688.4 | 0.4411764705882353 |
| static_threshold static | post-drift W2 seed 42 (test [818, 993], onset 1000) | 0.5833 | 0.7500 | 507.3 | 0.4411764705882353 |
| static_threshold static | post-drift W3 seed 42 (test [818, 993], onset 1000) | 0.4167 | 0.5000 | 552.4 | 0.4411764705882353 |
| risk_controlled static | post-drift W1 seed 43 (test [818, 993], onset 1000) | 0.2691 | 0.5833 | 1851.1 | 0.4838709677419355 |
| risk_controlled static | post-drift W2 seed 43 (test [818, 993], onset 1000) | 0.0000 | 0.3333 | 1366.4 | 0.4838709677419355 |
| risk_controlled static | post-drift W3 seed 43 (test [818, 993], onset 1000) | 0.0000 | 0.1667 | 729.0 | 0.4838709677419355 |
| risk_controlled online | post-drift W1 seed 43 (test [818, 993], onset 1000) | 0.1719 | 0.5000 | 1867.3 | 0.4838709677419355 |
| risk_controlled online | post-drift W2 seed 43 (test [818, 993], onset 1000) | 0.0000 | 0.3333 | 1366.4 | 0.4838709677419355 |
| risk_controlled online | post-drift W3 seed 43 (test [818, 993], onset 1000) | 0.0833 | 0.2500 | 704.3 | 0.4838709677419355 |
| point_forecast static | post-drift W1 seed 43 (test [818, 993], onset 1000) | 0.1722 | 0.4167 | 1792.1 | 0.45161290322580644 |
| point_forecast static | post-drift W2 seed 43 (test [818, 993], onset 1000) | 0.1667 | 0.3333 | 489.5 | 0.45161290322580644 |
| point_forecast static | post-drift W3 seed 43 (test [818, 993], onset 1000) | 0.0000 | 0.0000 | 0.0 | 0.45161290322580644 |
| static_threshold static | post-drift W1 seed 43 (test [818, 993], onset 1000) | 0.5875 | 0.7500 | 1658.6 | 0.5483870967741935 |
| static_threshold static | post-drift W2 seed 43 (test [818, 993], onset 1000) | 0.4167 | 0.5000 | 360.7 | 0.5483870967741935 |
| static_threshold static | post-drift W3 seed 43 (test [818, 993], onset 1000) | 0.2500 | 0.4167 | 628.8 | 0.5483870967741935 |
| risk_controlled static | post-drift W1 seed 44 (test [818, 993], onset 1000) | 0.2083 | 0.5833 | 1925.2 | 0.5357142857142857 |
| risk_controlled static | post-drift W2 seed 44 (test [818, 993], onset 1000) | 0.0000 | 0.0833 | 468.2 | 0.5357142857142857 |
| risk_controlled static | post-drift W3 seed 44 (test [818, 993], onset 1000) | 0.0000 | 0.3333 | 1042.6 | 0.5357142857142857 |
| risk_controlled online | post-drift W1 seed 44 (test [818, 993], onset 1000) | 0.1667 | 0.5000 | 1924.4 | 0.5357142857142857 |
| risk_controlled online | post-drift W2 seed 44 (test [818, 993], onset 1000) | 0.0000 | 0.0833 | 468.2 | 0.5357142857142857 |
| risk_controlled online | post-drift W3 seed 44 (test [818, 993], onset 1000) | 0.0417 | 0.3333 | 1032.2 | 0.5357142857142857 |
| point_forecast static | post-drift W1 seed 44 (test [818, 993], onset 1000) | 0.0476 | 0.5000 | 1932.9 | 0.4642857142857143 |
| point_forecast static | post-drift W2 seed 44 (test [818, 993], onset 1000) | 0.0417 | 0.0833 | 459.1 | 0.4642857142857143 |
| point_forecast static | post-drift W3 seed 44 (test [818, 993], onset 1000) | 0.0000 | 0.3333 | 865.4 | 0.4642857142857143 |
| static_threshold static | post-drift W1 seed 44 (test [818, 993], onset 1000) | 0.7465 | 0.9167 | 1616.9 | 0.6428571428571429 |
| static_threshold static | post-drift W2 seed 44 (test [818, 993], onset 1000) | 0.3750 | 0.4167 | 373.3 | 0.6428571428571429 |
| static_threshold static | post-drift W3 seed 44 (test [818, 993], onset 1000) | 0.1250 | 0.4167 | 1008.6 | 0.6428571428571429 |
| risk_controlled static | post-drift W1 seed 45 (test [818, 993], onset 1000) | 0.2139 | 0.5833 | 2583.0 | 0.5 |
| risk_controlled static | post-drift W2 seed 45 (test [818, 993], onset 1000) | 0.0000 | 0.2500 | 1340.0 | 0.5 |
| risk_controlled static | post-drift W3 seed 45 (test [818, 993], onset 1000) | 0.0000 | 0.0000 | 0.0 | 0.5 |
| risk_controlled online | post-drift W1 seed 45 (test [818, 993], onset 1000) | 0.1722 | 0.5833 | 2591.4 | 0.5 |
| risk_controlled online | post-drift W2 seed 45 (test [818, 993], onset 1000) | 0.0417 | 0.2500 | 1334.2 | 0.5 |
| risk_controlled online | post-drift W3 seed 45 (test [818, 993], onset 1000) | 0.0833 | 0.0833 | -11.7 | 0.5 |
| point_forecast static | post-drift W1 seed 45 (test [818, 993], onset 1000) | 0.0897 | 0.5833 | 2599.1 | 0.42857142857142855 |
| point_forecast static | post-drift W2 seed 45 (test [818, 993], onset 1000) | 0.1667 | 0.4167 | 1329.7 | 0.42857142857142855 |
| point_forecast static | post-drift W3 seed 45 (test [818, 993], onset 1000) | 0.0000 | 0.0000 | 0.0 | 0.42857142857142855 |
| static_threshold static | post-drift W1 seed 45 (test [818, 993], onset 1000) | 0.7415 | 0.9167 | 2191.5 | 0.5357142857142857 |
| static_threshold static | post-drift W2 seed 45 (test [818, 993], onset 1000) | 0.2639 | 0.5000 | 1450.5 | 0.5357142857142857 |
| static_threshold static | post-drift W3 seed 45 (test [818, 993], onset 1000) | 0.1667 | 0.1667 | -40.6 | 0.5357142857142857 |
| risk_controlled static | post-drift W1 seed 46 (test [818, 993], onset 1000) | 0.2647 | 0.5833 | 1757.4 | 0.46875 |
| risk_controlled static | post-drift W2 seed 46 (test [818, 993], onset 1000) | 0.0000 | 0.4167 | 1971.0 | 0.46875 |
| risk_controlled static | post-drift W3 seed 46 (test [818, 993], onset 1000) | 0.0000 | 0.2500 | 836.4 | 0.46875 |
| risk_controlled online | post-drift W1 seed 46 (test [818, 993], onset 1000) | 0.0980 | 0.4167 | 1767.7 | 0.46875 |
| risk_controlled online | post-drift W2 seed 46 (test [818, 993], onset 1000) | 0.0833 | 0.5000 | 1969.8 | 0.46875 |
| risk_controlled online | post-drift W3 seed 46 (test [818, 993], onset 1000) | 0.0833 | 0.3333 | 822.0 | 0.46875 |
| point_forecast static | post-drift W1 seed 46 (test [818, 993], onset 1000) | 0.0952 | 0.4167 | 1774.2 | 0.40625 |
| point_forecast static | post-drift W2 seed 46 (test [818, 993], onset 1000) | 0.0000 | 0.4167 | 1971.0 | 0.40625 |
| point_forecast static | post-drift W3 seed 46 (test [818, 993], onset 1000) | 0.0000 | 0.2500 | 836.4 | 0.40625 |
| static_threshold static | post-drift W1 seed 46 (test [818, 993], onset 1000) | 0.7234 | 0.9167 | 1681.1 | 0.5 |
| static_threshold static | post-drift W2 seed 46 (test [818, 993], onset 1000) | 0.5417 | 0.8333 | 1125.2 | 0.5 |
| static_threshold static | post-drift W3 seed 46 (test [818, 993], onset 1000) | 0.4167 | 0.6667 | 736.9 | 0.5 |

## Paired statistics (proposed static alpha=0.10 vs baselines, all seeds)

| Scenario | Comparison | Metric | Direction | Wilcoxon p | Bootstrap 95% CI (proposed - other) |
|---|---|---|---|---:|---|
| noisy_bursty | vs point_forecast | wasted | proposed higher | 0.18750 | [0.4000, 4.0000] |
| noisy_bursty | vs static_threshold | wasted | proposed lower | 0.06250 | [-59.4000, -54.0000] |
| noisy_bursty | vs point_forecast | net_benefit | proposed lower | 0.62500 | [-414.6880, 332.8160] |
| noisy_bursty | vs static_threshold | net_benefit | proposed higher | 0.06250 | [9719.5680, 11588.0920] |
| noisy_bursty | vs point_forecast | recall | proposed higher | 0.12500 | [0.0310, 0.1257] |
| noisy_bursty | vs static_threshold | recall | proposed lower | 0.87500 | [-0.1147, 0.0647] |
| stress | vs point_forecast | wasted | proposed higher | 1.00000 | [0.0000, 2.2000] |
| stress | vs static_threshold | wasted | proposed lower | 0.06250 | [-39.8000, -33.0000] |
| stress | vs point_forecast | net_benefit | proposed higher | 0.31250 | [347.0360, 13245.5920] |
| stress | vs static_threshold | net_benefit | proposed higher | 0.06250 | [6093.7680, 13981.1280] |
| stress | vs point_forecast | recall | proposed higher | 0.06250 | [0.0454, 0.0696] |
| stress | vs static_threshold | recall | proposed lower | 0.12500 | [-0.0816, -0.0196] |
| simpy | vs point_forecast | p99_latency | proposed higher | 0.75000 | [-2897.1880, 13589.1280] |
| simpy | vs static_threshold | p99_latency | proposed higher | 0.25000 | [1953.1240, 31528.7610] |

Where the proposed method loses: noisy_bursty: more wasted migrations than point_forecast; noisy_bursty: lower net benefit than point_forecast; noisy_bursty: lower recall than static_threshold; stress: more wasted migrations than point_forecast; stress: lower recall than static_threshold; simpy: higher p99 latency than point_forecast; simpy: higher p99 latency than static_threshold

## SimPy capped slices

| Seed | Policy | Mean lat | p95 | p99 | IOPS | Throughput | SSD hit | SLA viol | HDD service ms | n_requests |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | hdd_only | 53139.069 | 104296.343 | 117057.933 | 11.0 | 11.0 | 0.0000 | 0.9990 | 50.0000 | 50000 |
| 42 | all_ssd | 0.100 | 0.299 | 0.457 | 11.0 | 11.0 | 1.0000 | 0.0000 | 50.0000 | 50000 |
| 42 | risk_controlled | 51267.724 | 102246.511 | 114795.894 | 11.0 | 11.0 | 0.0135 | 0.9855 | 50.0000 | 50000 |
| 42 | point_forecast | 51003.220 | 101824.350 | 114628.709 | 11.0 | 11.0 | 0.0159 | 0.9831 | 50.0000 | 50000 |
| 42 | static_threshold | 40509.238 | 88474.690 | 99971.415 | 11.0 | 11.0 | 0.0978 | 0.9011 | 50.0000 | 50000 |
| 43 | hdd_only | 54031.574 | 109074.442 | 122878.196 | 11.6 | 11.6 | 0.0000 | 0.9997 | 50.0000 | 50000 |
| 43 | all_ssd | 0.100 | 0.298 | 0.457 | 11.7 | 11.7 | 1.0000 | 0.0000 | 50.0000 | 50000 |
| 43 | risk_controlled | 50544.575 | 104402.976 | 118286.368 | 11.6 | 11.6 | 0.0260 | 0.9736 | 50.0000 | 50000 |
| 43 | point_forecast | 38866.350 | 89134.781 | 104697.240 | 11.7 | 11.7 | 0.1197 | 0.8799 | 50.0000 | 50000 |
| 43 | static_threshold | 29457.289 | 74415.988 | 86757.607 | 11.7 | 11.7 | 0.2014 | 0.7983 | 50.0000 | 50000 |
| 44 | hdd_only | 54787.452 | 112828.284 | 125230.644 | 11.6 | 11.6 | 0.0000 | 0.9999 | 50.0000 | 50000 |
| 44 | all_ssd | 0.100 | 0.299 | 0.457 | 11.7 | 11.7 | 1.0000 | 0.0000 | 50.0000 | 50000 |
| 44 | risk_controlled | 50504.625 | 107981.802 | 120438.234 | 11.6 | 11.6 | 0.0322 | 0.9677 | 50.0000 | 50000 |
| 44 | point_forecast | 53296.904 | 110840.672 | 123335.421 | 11.6 | 11.6 | 0.0109 | 0.9890 | 50.0000 | 50000 |
| 44 | static_threshold | 49022.230 | 106034.591 | 118485.109 | 11.6 | 11.6 | 0.0429 | 0.9570 | 50.0000 | 50000 |

Target HDD utilisation rho in [0.6, 0.8]. M/M/1 relative error: 0.0516.

### Matched-volume sweep

| Seed | Setting | p99 | hit | SLA |
|---:|---|---:|---:|---:|
| 42 | point_forecast_vol0.9 | 115961.562 | 0.0051 | 0.9939 |
| 42 | point_forecast_vol1.0 | 113856.838 | 0.0211 | 0.9779 |
| 42 | point_forecast_vol1.1 | 112451.059 | 0.0220 | 0.9769 |
| 42 | static_threshold_vol0.9 | 108137.246 | 0.0537 | 0.9453 |
| 42 | static_threshold_vol1.0 | 114044.275 | 0.0183 | 0.9806 |
| 42 | static_threshold_vol1.1 | 114621.587 | 0.0170 | 0.9820 |
| 43 | point_forecast_vol0.9 | 122013.707 | 0.0064 | 0.9932 |
| 43 | point_forecast_vol1.0 | 121530.146 | 0.0060 | 0.9937 |
| 43 | point_forecast_vol1.1 | 120197.037 | 0.0161 | 0.9835 |
| 43 | static_threshold_vol0.9 | 113175.416 | 0.0577 | 0.9420 |
| 43 | static_threshold_vol1.0 | 119899.841 | 0.0147 | 0.9849 |
| 43 | static_threshold_vol1.1 | 119874.641 | 0.0152 | 0.9845 |
| 44 | point_forecast_vol0.9 | 123290.632 | 0.0109 | 0.9890 |
| 44 | point_forecast_vol1.0 | 122890.984 | 0.0111 | 0.9888 |
| 44 | point_forecast_vol1.1 | 123391.995 | 0.0094 | 0.9905 |
| 44 | static_threshold_vol0.9 | 112119.800 | 0.0876 | 0.9123 |
| 44 | static_threshold_vol1.0 | 122058.509 | 0.0193 | 0.9806 |
| 44 | static_threshold_vol1.1 | 121414.075 | 0.0206 | 0.9793 |

## MSRC hm_0 and prn_0

### hm_0

Header/columns: ['timestamp_s', 'extent', 'is_read', 'size_bytes', 'response_ms', 'DiskNumber']
Units: {'timestamp': 'seconds (converted from Windows filetime in loader)', 'extent': 'Offset // 16777216 bytes', 'size': 'bytes'}
Row count: 3993315
First 5 rows: `[{'timestamp_s': 1172164929.551488, 'extent': 706, 'is_read': 0, 'size_bytes': 512, 'response_ms': 0.2617, 'DiskNumber': 0}, {'timestamp_s': 1172164929.5515168, 'extent': 330, 'is_read': 0, 'size_bytes': 4096, 'response_ms': 0.232, 'DiskNumber': 0}, {'timestamp_s': 1172164930.145152, 'extent': 188, 'is_read': 0, 'size_bytes': 40960, 'response_ms': 0.3438, 'DiskNumber': 0}, {'timestamp_s': 1172164930.145392, 'extent': 188, 'is_read': 0, 'size_bytes': 4096, 'response_ms': 0.1037, 'DiskNumber': 0}, {'timestamp_s': 1172164930.145408, 'extent': 188, 'is_read': 0, 'size_bytes': 4096, 'response_ms': 0.0883, 'DiskNumber': 0}]`
Prevalence Y>=tau: 0.019630162906229032 ; median tau: 32.8506329113924
Notes: ['sliced to first 1500 of 10080 windows for tractable training', 'prevalence Y>=tau is 0.0196 with unchanged tau=32.8506; did not lower tau. Evaluation still runs on this volume.']

| Policy | Migrations | FDP | Activity | Net benefit | Recall |
|---|---:|---:|---:|---:|---:|
| risk_controlled | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| point_forecast | 13 | 0.0536 | 0.1607 | 3309.64 | 0.056818181818181816 |
| static_threshold | 73 | 0.3244 | 0.4286 | -10049.56 | 0.07954545454545454 |
| hdd_only | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| all_ssd | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| oracle | 88 | 0.0 | 0.3036 | 86616.74 | 1.0 |

### prn_0

Header/columns: ['timestamp_s', 'extent', 'is_read', 'size_bytes', 'response_ms', 'DiskNumber']
Units: {'timestamp': 'seconds (converted from Windows filetime in loader)', 'extent': 'Offset // 16777216 bytes', 'size': 'bytes'}
Row count: 5585885
First 5 rows: `[{'timestamp_s': 1172163600.3457265, 'extent': 188, 'is_read': 0, 'size_bytes': 4096, 'response_ms': 0.1257, 'DiskNumber': 0}, {'timestamp_s': 1172163600.3457265, 'extent': 193, 'is_read': 0, 'size_bytes': 4096, 'response_ms': 0.1271, 'DiskNumber': 0}, {'timestamp_s': 1172163601.34572, 'extent': 145, 'is_read': 0, 'size_bytes': 4096, 'response_ms': 0.2151, 'DiskNumber': 0}, {'timestamp_s': 1172163604.001952, 'extent': 3543, 'is_read': 0, 'size_bytes': 512, 'response_ms': 0.1653, 'DiskNumber': 0}, {'timestamp_s': 1172163604.001952, 'extent': 3543, 'is_read': 0, 'size_bytes': 512, 'response_ms': 0.122, 'DiskNumber': 0}]`
Prevalence Y>=tau: 0.036380580915727524 ; median tau: 32.8506329113924
Notes: ['sliced to first 1500 of 10080 windows for tractable training']

| Policy | Migrations | FDP | Activity | Net benefit | Recall |
|---|---:|---:|---:|---:|---:|
| risk_controlled | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| point_forecast | 20 | 0.0238 | 0.2143 | 35384.0 | 0.04225352112676056 |
| static_threshold | 144 | 0.4587 | 0.6071 | -896.58 | 0.13380281690140844 |
| hdd_only | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| all_ssd | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| oracle | 142 | 0.0 | 0.6964 | 79744.26 | 1.0 |

