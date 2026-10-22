"""LTT family Bonferroni and nested-selection tests."""
from __future__ import annotations

import numpy as np

from src.calibration.ltt_gate import (
    _nested_select,
    compute_certifiability,
    extent_shard_ids,
    make_unit_ids,
    run_ltt_certification,
)
from src.simulator.cost_model import CostModel, CostModelConfig


class DummyCost:
    def compute_realized_net_benefit(self, y: float) -> float:
        return float(y - 5.0)


def test_bonferroni_over_families_not_quantiles():
    n = 200
    rng = np.random.default_rng(0)
    Y = rng.gamma(2.0, 8.0, size=n)
    tau = np.full(n, 5.0)
    epoch = np.repeat(np.arange(20), 10)
    extent = np.arange(n) % 40
    q_preds = {q: Y * (0.4 + q) for q in [0.05, 0.10, 0.20, 0.30, 0.50]}
    p_hat = 1.0 / (1.0 + np.exp(-(Y - 8.0) / 5.0))
    cm = DummyCost()
    result = run_ltt_certification(
        q_preds, tau, Y, epoch, cm, [0.05, 0.10, 0.20, 0.30, 0.50],
        alpha_fmr=0.10, delta_ltt=0.10, verbose=False, p_hat=p_hat,
        extent_ids=extent, n_extent_shards=8,
    )
    assert result.n_families_tested == 2
    assert abs(result.p_threshold - 0.05) < 1e-9
    assert result.delta_ltt == 0.10
    assert result.K == 2


def test_shard_hash_is_stable():
    ids = np.array([0, 1, 7, 8, 15, 16], dtype=np.int64)
    a = extent_shard_ids(ids, 8)
    b = extent_shard_ids(ids, 8)
    assert np.array_equal(a, b)
    assert set(a.tolist()).issubset(set(range(8)))


def test_units_increase_with_shards():
    epoch = np.repeat(np.arange(10), 16)
    extent = np.tile(np.arange(16), 10)
    u1, n1, n_ep = make_unit_ids(epoch, None, n_extent_shards=1)
    u8, n8, _ = make_unit_ids(epoch, extent, n_extent_shards=8)
    assert n_ep == 10
    assert n8 >= n1


def test_nested_select_least_conservative_then_net_benefit():
    from src.calibration.ltt_gate import GateCell
    cells = []
    for delta in [2.0, 1.0, 0.0, -0.5]:
        cells.append(GateCell(
            q=0.1, delta=delta, fdp_values=np.array([0.0]),
            n_epochs_with_mig=1, total_migrations=10, wasted_migrations=0,
            mean_fdp=0.0, pooled_fmr=0.0, p_value=0.01, certified=True,
            net_benefit=10.0 - delta, migration_activity=0.5, family="quantile",
        ))
    best = _nested_select(cells)
    assert best is not None
    assert best.delta == -0.5


def test_certifiability_uses_family_count():
    a_k2, _ = compute_certifiability(400, 2, 0.10)
    a_k9, _ = compute_certifiability(400, 9, 0.10)
    assert a_k2 < a_k9
