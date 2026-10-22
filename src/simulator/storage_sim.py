"""
SimPy-based discrete-event queue simulator for two-tier storage (HDD + SSD).

Used for:
- M/M/1 queueing theory validation test
- Capped-slice headline latency evaluations
- Simulating foreground I/O interference (kappa) during background migrations
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import simpy


@dataclass
class SimConfig:
    hdd_capacity: int = 1
    ssd_capacity: int = 4
    mean_service_hdd_ms: float = 8.0
    mean_service_ssd_ms: float = 0.1
    kappa_interference: float = 1.0
    extent_size_mib: float = 1.0


class StorageSim:
    """Discrete-event queue simulator using SimPy."""

    def __init__(self, config: Optional[SimConfig] = None) -> None:
        self.cfg = config or SimConfig()

    def run_mm1_verification(
        self,
        arrival_rate_lambda: float,
        service_rate_mu: float,
        n_requests: int = 10000,
        seed: int = 42,
    ) -> Tuple[float, float, float]:
        """
        Validate simulator against theoretical M/M/1 queueing formula:
            W_theory = 1.0 / (mu - lambda)
        Returns:
            (simulated_mean_latency, theoretical_mean_latency, relative_error)
        """
        if arrival_rate_lambda >= service_rate_mu:
            raise ValueError("System is unstable: arrival rate lambda >= service rate mu")

        rng = np.random.default_rng(seed)
        env = simpy.Environment()
        server = simpy.Resource(env, capacity=1)

        latencies: List[float] = []

        def request_process(arrival_time: float):
            with server.request() as req:
                yield req
                # Service time Exp(mu)
                service_time = float(rng.exponential(1.0 / service_rate_mu))
                yield env.timeout(service_time)
                departure_time = env.now
                latencies.append(departure_time - arrival_time)

        def source():
            for _ in range(n_requests):
                inter_arrival = float(rng.exponential(1.0 / arrival_rate_lambda))
                yield env.timeout(inter_arrival)
                env.process(request_process(env.now))

        env.process(source())
        env.run()

        # Warmup trim: discard first 10%
        valid_latencies = latencies[int(n_requests * 0.1):]
        sim_mean = float(np.mean(valid_latencies))
        theory_mean = 1.0 / (service_rate_mu - arrival_rate_lambda)
        rel_error = abs(sim_mean - theory_mean) / theory_mean

        return sim_mean, theory_mean, rel_error

    def simulate_trace_slice(
        self,
        requests_df: np.ndarray, # array of (timestamp_ms, extent, is_read, size_kb)
        ssd_extents: set[int],
        seed: int = 42,
    ) -> Dict[str, float]:
        """
        Simulate a time slice of requests on HDD and SSD resources.
        Returns mean latency, p95 latency, HDD utilization, SSD utilization.
        """
        rng = np.random.default_rng(seed)
        env = simpy.Environment()
        hdd = simpy.Resource(env, capacity=self.cfg.hdd_capacity)
        ssd = simpy.Resource(env, capacity=self.cfg.ssd_capacity)

        latencies: List[float] = []
        hdd_busy_time = 0.0
        ssd_busy_time = 0.0

        def handle_request(req_extent: int):
            nonlocal hdd_busy_time, ssd_busy_time
            t_start = env.now
            if req_extent in ssd_extents:
                with ssd.request() as req:
                    yield req
                    st = float(rng.exponential(self.cfg.mean_service_ssd_ms))
                    ssd_busy_time += st
                    yield env.timeout(st)
            else:
                with hdd.request() as req:
                    yield req
                    st = float(rng.exponential(self.cfg.mean_service_hdd_ms))
                    hdd_busy_time += st
                    yield env.timeout(st)
            latencies.append(env.now - t_start)

        def request_generator():
            last_t = 0.0
            for r in requests_df:
                t_arr = float(r[0])
                delay = max(0.0, t_arr - last_t)
                if delay > 0:
                    yield env.timeout(delay)
                last_t = t_arr
                env.process(handle_request(int(r[1])))

        env.process(request_generator())
        env.run()

        total_time = max(env.now, 1.0)
        step = max(1, len(latencies) // 500) if latencies else 1
        return {
            "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else 0.0,
            "iops": float(len(latencies) / (total_time / 1000.0)),
            "throughput_iops": float(len(latencies) / (total_time / 1000.0)),
            "ssd_hit_ratio": float(sum(1 for r in requests_df if int(r[1]) in ssd_extents) / max(1, len(requests_df))),
            "sla_violation_rate": float(np.mean(np.asarray(latencies) > 10.0)) if latencies else 0.0,
            "hdd_utilization": float(hdd_busy_time / (total_time * self.cfg.hdd_capacity)),
            "ssd_utilization": float(ssd_busy_time / (total_time * self.cfg.ssd_capacity)),
            "latency_samples": [float(x) for x in latencies[::step][:500]],
        }
