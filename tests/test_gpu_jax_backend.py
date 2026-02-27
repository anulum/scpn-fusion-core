# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — JAX GPU Backend Parity Tests
# ──────────────────────────────────────────────────────────────────────
"""Verify JAX multigrid produces results within tolerance of CPU baseline."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.gpu_runtime import GPURuntimeBridge

try:
    import jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


@pytest.fixture
def bridge():
    return GPURuntimeBridge(seed=42)


@pytest.fixture
def test_field():
    return np.linspace(0.0, 1.0, 64 * 64).reshape(64, 64)


class TestJAXBackendAvailability:
    def test_jax_in_available_backends(self, bridge):
        backends = bridge.available_equilibrium_backends()
        if HAS_JAX:
            assert "jax" in backends
        else:
            assert "jax" not in backends


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestJAXMultigridParity:
    def test_jax_matches_gpu_sim(self, bridge, test_field):
        cpu_result = bridge._gpu_sim_multigrid(test_field, iterations=4)
        jax_result = bridge._jax_multigrid(test_field, iterations=4)
        np.testing.assert_allclose(jax_result, cpu_result, atol=1e-10)

    def test_jax_smooths_field(self, bridge, test_field):
        result = bridge._jax_multigrid(test_field, iterations=4)
        # Smoothing must reduce variance compared to input
        assert np.var(result) < np.var(test_field)
        assert np.all(np.isfinite(result))

    def test_jax_single_iteration(self, bridge, test_field):
        result = bridge._jax_multigrid(test_field, iterations=1)
        assert result.shape == test_field.shape
        assert np.all(np.isfinite(result))

    def test_jax_larger_grid(self, bridge):
        field = np.random.default_rng(99).uniform(0, 1, (128, 128))
        cpu = bridge._gpu_sim_multigrid(field, iterations=3)
        jax_out = bridge._jax_multigrid(field, iterations=3)
        np.testing.assert_allclose(jax_out, cpu, atol=1e-10)

    def test_jax_benchmark_latency(self, bridge):
        bench = bridge.benchmark_equilibrium_latency(
            backend="jax", trials=16, grid_size=32,
        )
        assert bench.backend == "jax"
        assert bench.mean_ms_wall > 0.0
        assert bench.trials == 16
