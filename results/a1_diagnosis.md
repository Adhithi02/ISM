# A1 Gate diagnosis (current LTT, noisy_bursty seed 42)

LTT certifies a (q, delta) cell on Cal-B, but the test policy always applies delta to q10 instead of the certified quantile, so the deployed gate is far more conservative than the certified cell; Bonferroni over every quantile and mean FDP on inactive units further favours quiet cells.

## alpha = 0.05

| Quantity | Value |
|---|---|
| Calibration units (epoch x shard) | 480 |
| Calibration epochs | 60 |
| Certified cells | 119 |
| Migrating units in chosen cell | 469 |
| Smallest p-value reached | 0.0 |
| Chosen (q, delta) | (0.3, -0.2) |
| Chosen p-value | 0.012433 |
| Chosen certified | True |
| Chosen Cal-B activity | 0.9771 |
| Chosen Cal-B migrations | 706 |
| Test migrations if always q10 | 0 |
| Test migrations if matched q | 17 |

## alpha = 0.10

| Quantity | Value |
|---|---|
| Calibration units (epoch x shard) | 480 |
| Calibration epochs | 60 |
| Certified cells | 126 |
| Migrating units in chosen cell | 480 |
| Smallest p-value reached | 0.0 |
| Chosen (q, delta) | (0.5, -0.2) |
| Chosen p-value | 0.006675 |
| Chosen certified | True |
| Chosen Cal-B activity | 1.0 |
| Chosen Cal-B migrations | 886 |
| Test migrations if always q10 | 0 |
| Test migrations if matched q | 20 |

## alpha = 0.20

| Quantity | Value |
|---|---|
| Calibration units (epoch x shard) | 480 |
| Calibration epochs | 60 |
| Certified cells | 128 |
| Migrating units in chosen cell | 480 |
| Smallest p-value reached | 0.0 |
| Chosen (q, delta) | (0.5, -0.3) |
| Chosen p-value | 1e-06 |
| Chosen certified | True |
| Chosen Cal-B activity | 1.0 |
| Chosen Cal-B migrations | 1007 |
| Test migrations if always q10 | 2 |
| Test migrations if matched q | 23 |

## alpha = 0.30

| Quantity | Value |
|---|---|
| Calibration units (epoch x shard) | 480 |
| Calibration epochs | 60 |
| Certified cells | 129 |
| Migrating units in chosen cell | 480 |
| Smallest p-value reached | 0.0 |
| Chosen (q, delta) | (0.5, -0.3) |
| Chosen p-value | 0.0 |
| Chosen certified | True |
| Chosen Cal-B activity | 1.0 |
| Chosen Cal-B migrations | 1007 |
| Test migrations if always q10 | 2 |
| Test migrations if matched q | 23 |

Point-forecast test migrations: 17

Dominant cause: certified (q, delta) is not the score applied at test time (always q10), so the deployed policy is nearly silent.
