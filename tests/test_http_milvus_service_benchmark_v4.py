"""Unit tests for the service benchmark's deterministic scoring helpers."""

from __future__ import annotations

import unittest

from scripts.run_http_milvus_service_benchmark_v4 import _performance_gates, _stats


class ServiceBenchmarkV4Tests(unittest.TestCase):
    def test_stats_uses_explicit_support_and_nearest_rank_p95(self) -> None:
        observed = _stats([8.0, 1.0, 3.0, 2.0, 5.0, 7.0, 4.0, 6.0])
        self.assertEqual(observed["support"], 8)
        self.assertEqual(observed["p50"], 4.5)
        self.assertEqual(observed["p95"], 8.0)

    def test_performance_gate_compares_candidate_to_fixed_baseline(self) -> None:
        roles = {
            "current_system_repair_checkpoint_87": {
                "steady": {"1": {"failure_rate": 0.0, "stage_latency_ms": {"http_e2e_ms": {"p95": 100.0}}}}
            },
            "targeted_exploration_adapter_v4": {
                "steady": {"1": {"failure_rate": 0.0, "stage_latency_ms": {"http_e2e_ms": {"p95": 120.0}}}}
            },
        }
        gate = _performance_gates(
            roles,
            {
                "failure_rate_max": 0.02,
                "candidate_to_checkpoint_87_concurrency_1_p95_ratio_max": 1.25,
            },
        )
        self.assertEqual(gate["status"], "PASS")
        self.assertAlmostEqual(
            gate["candidate_to_baseline_concurrency_1_http_p95"]["observed_ratio"], 1.2
        )

    def test_performance_gate_keeps_latency_regression_negative(self) -> None:
        roles = {
            "current_system_repair_checkpoint_87": {
                "steady": {"1": {"failure_rate": 0.0, "stage_latency_ms": {"http_e2e_ms": {"p95": 100.0}}}}
            },
            "targeted_exploration_adapter_v4": {
                "steady": {"1": {"failure_rate": 0.0, "stage_latency_ms": {"http_e2e_ms": {"p95": 126.0}}}}
            },
        }
        gate = _performance_gates(
            roles,
            {
                "failure_rate_max": 0.02,
                "candidate_to_checkpoint_87_concurrency_1_p95_ratio_max": 1.25,
            },
        )
        self.assertEqual(gate["status"], "FAIL")

    def test_performance_gate_accepts_configured_role_names(self) -> None:
        roles = {
            "v4": {"steady": {"1": {"failure_rate": 0.0, "stage_latency_ms": {"http_e2e_ms": {"p95": 100.0}}}}},
            "v5": {"steady": {"1": {"failure_rate": 0.0, "stage_latency_ms": {"http_e2e_ms": {"p95": 110.0}}}}},
        }
        gate = _performance_gates(roles, {
            "baseline_role": "v4",
            "candidate_role": "v5",
            "failure_rate_max": 0.02,
            "candidate_to_checkpoint_87_concurrency_1_p95_ratio_max": 1.25,
        })
        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["baseline_role"], "v4")
        self.assertEqual(gate["candidate_role"], "v5")


if __name__ == "__main__":
    unittest.main()
