"""
Tests for simulator, queueing theory, cost model, eviction, and fast replay.

Tests:
- test_mm1_theory: verifies SimPy queue against theoretical M/M/1 formula (< 10% error).
- test_lru_eviction: verifies LRU evicts the least recently accessed extent.
- test_value_eviction: verifies ValueEviction evicts lowest predicted value.
- test_cost_model_tau: verifies state-dependent tau computation.
- test_cost_model_rescale: verifies rescale factor applied when median tau < 5.0.
- test_fast_replay_budget: verifies migrations per epoch <= max budget M.
- test_fast_replay_fdp_zero_when_no_migrations: verifies FDP = 0 when nothing migrates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.simulator.storage_sim import StorageSim, SimConfig
from src.simulator.eviction import LRUEviction, ValueEviction
from src.simulator.cost_model import CostModel, CostModelConfig
from src.simulator.fast_replay import run_fast_replay


class TestStorageSimulator:
    """Queueing theory and simulator tests."""

    def test_mm1_theory(self):
        """M/M/1 verification: simulated mean latency matches 1/(mu - lambda) within 10%."""
        sim = StorageSim()
        arrival_rate = 0.07   # lambda = 70 req/s = 0.07 req/ms
        service_rate = 0.10   # mu = 100 req/s = 0.10 req/ms -> rho = 0.70 (target rho range 0.60 - 0.80)

        sim_mean, theory_mean, rel_error = sim.run_mm1_verification(
            arrival_rate_lambda=arrival_rate,
            service_rate_mu=service_rate,
            n_requests=10000,
            seed=42,
        )

        assert rel_error < 0.10, (
            f"M/M/1 theory relative error {rel_error*100:.2f}% >= 10%. "
            f"Sim={sim_mean:.2f} ms, Theory={theory_mean:.2f} ms"
        )


class TestEvictionPolicies:
    """Tests for pluggable eviction implementations."""

    def test_lru_eviction(self):
        """LRU must evict the extent with oldest timestamp."""
        lru = LRUEviction()
        lru.add_extent(10, timestamp=100.0)
        lru.add_extent(20, timestamp=105.0)
        lru.add_extent(30, timestamp=110.0)

        # Access extent 10 to make 20 the oldest
        lru.record_access(10, timestamp=120.0)

        evictee = lru.select_evictee({10, 20, 30})
        assert evictee == 20, f"Expected LRU evictee 20, got {evictee}"

    def test_value_eviction(self):
        """ValueEviction must evict extent with lowest predicted future accesses."""
        val_evict = ValueEviction()
        val_evict.add_extent(10, timestamp=100.0, predicted_value=25.0)
        val_evict.add_extent(20, timestamp=105.0, predicted_value=3.5)
        val_evict.add_extent(30, timestamp=110.0, predicted_value=12.0)

        evictee = val_evict.select_evictee({10, 20, 30})
        assert evictee == 20, f"Expected lowest-value evictee 20, got {evictee}"


class TestCostModel:
    """Tests for state-dependent tau and rescaling."""

    def test_cost_model_tau(self):
        """Tau with free capacity vs with eviction value."""
        cm = CostModel(CostModelConfig(l_hdd_ms=8.0, l_ssd_ms=0.1, dirty_fraction=0.3))
        # Delta_L = 7.9, C_mig = 8.1, C_wb = 2.4
        # When free: tau = (8.1 + 2.4) / 7.9 = 10.5 / 7.9 = 1.329
        tau_free = cm.compute_tau(evictee_pred_q50=None)
        assert abs(tau_free - (10.5 / 7.9)) < 0.01

        # With eviction value q50 = 4.0: V_evict = 4.0 * 7.9 = 31.6
        # Total cost = 10.5 + 31.6 = 42.1 -> tau = 42.1 / 7.9 = 5.329
        tau_evict = cm.compute_tau(evictee_pred_q50=4.0)
        assert abs(tau_evict - (42.1 / 7.9)) < 0.01
        assert tau_evict > tau_free

    def test_cost_model_rescale(self):
        """If median tau < 5.0, rescale_factor increases so median >= 5.0."""
        cm = CostModel(CostModelConfig(l_hdd_ms=8.0, l_ssd_ms=0.1, dirty_fraction=0.3, min_median_tau=5.0))
        # Evictees with low values: median tau will be < 5.0
        sample_q50s = [0.0, 0.5, 1.0, 1.2, 0.8]
        stats = cm.check_and_rescale_tau_distribution(sample_q50s)

        assert stats["final_median_tau"] >= 5.0, f"Rescaled median {stats['final_median_tau']} < 5.0"
        assert cm.rescale_factor > 1.0


class TestFastReplay:
    """Tests for fast trace replay mechanics."""

    def test_fast_replay_budget_and_fdp_zero(self):
        """Budget must never be exceeded and FDP must be 0 when nothing migrates."""
        # Create minimal synthetic decision epochs
        data = []
        for w in [5, 10, 15, 20]:
            for e in range(10):
                # Extent 0 is hot, others cold
                y = 15.0 if e == 0 else 0.0
                q_lo = 12.0 if e == 0 else 0.0
                data.append({"window": w, "extent": e, "Y": y, "q_lo": q_lo, "q50": y})
        df = pd.DataFrame(data)

        cm = CostModel(CostModelConfig(l_hdd_ms=8.0, l_ssd_ms=0.1, dirty_fraction=0.3))
        eviction = LRUEviction()
        max_budget = 2

        metrics = run_fast_replay(
            decision_df=df,
            ssd_capacity=5,
            max_migrations_per_epoch=max_budget,
            cost_model=cm,
            eviction_policy=eviction,
        )

        for m_t in metrics.epoch_migrations:
            assert m_t <= max_budget, f"Migrations {m_t} exceeded budget {max_budget}"

        # In epochs where m_t == 0, fdp must be 0.0
        for fdp, m_t in zip(metrics.epoch_fdp_series, metrics.epoch_migrations):
            if m_t == 0:
                assert fdp == 0.0, f"FDP was {fdp} when nothing migrated"

    def test_policies_produce_distinct_decision_logs(self):
        """LRU promotion and threshold promotion must not share a log."""
        data = []
        for w in [5, 10, 15]:
            for e in range(4):
                data.append({
                    "window": w, "extent": e, "Y": 8.0,
                    "q_lo": 8.0, "q50": 8.0,
                    "count_lag1": 1.0 if e == 0 else 6.0,
                })
        df = pd.DataFrame(data)
        cm = CostModel(CostModelConfig())
        from src.simulator.baselines import run_baseline
        lru = run_baseline("lru_promote", df, 4, 4, cm)
        threshold = run_baseline("static_threshold", df, 4, 4, cm, threshold=5.0)
        assert lru.decision_log != threshold.decision_log

    def test_policy_extremes_have_expected_hit_ratios(self):
        """HDD-only and all-SSD policy states are unambiguous."""
        from src.simulator.storage_sim import StorageSim, SimConfig

        requests = np.array([
            [0.0, 1.0, 1.0, 64.0],
            [1.0, 2.0, 1.0, 64.0],
            [2.0, 1.0, 1.0, 64.0],
        ])
        sim = StorageSim(SimConfig())
        hdd = sim.simulate_trace_slice(requests, set(), seed=1)
        ssd = sim.simulate_trace_slice(requests, {1, 2}, seed=1)
        assert hdd["ssd_hit_ratio"] == 0.0
        assert ssd["ssd_hit_ratio"] == 1.0
