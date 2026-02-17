#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — SNN Controller PyO3 Bridge Tests
# © 1998–2026 Miroslav Sotek. All rights reserved.
# ──────────────────────────────────────────────────────────────────────
"""Tests for the Rust SNN controller exposed via PyO3."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "src"))

try:
    import scpn_fusion_rs  # type: ignore[import-untyped]

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust extension not available")


# ─── PySnnPool tests ───


class TestPySnnPool:
    def test_positive_error_positive_output(self) -> None:
        """Sustained positive error should produce positive control output."""
        pool = scpn_fusion_rs.PySnnPool(n_neurons=50, gain=10.0, window_size=20)
        output = 0.0
        for _ in range(50):
            output = pool.step(5.0)
        assert output > 0.0, f"Expected positive output, got {output}"

    def test_negative_error_negative_output(self) -> None:
        """Sustained negative error should produce negative control output."""
        pool = scpn_fusion_rs.PySnnPool(n_neurons=50, gain=10.0, window_size=20)
        output = 0.0
        for _ in range(50):
            output = pool.step(-5.0)
        assert output < 0.0, f"Expected negative output, got {output}"

    def test_default_constructor(self) -> None:
        """Default constructor should use n=50, gain=10, window=20."""
        pool = scpn_fusion_rs.PySnnPool()
        assert pool.n_neurons == 50
        assert pool.gain == 10.0

    def test_getters(self) -> None:
        """Property getters should reflect constructor arguments."""
        pool = scpn_fusion_rs.PySnnPool(n_neurons=30, gain=5.0, window_size=10)
        assert pool.n_neurons == 30
        assert pool.gain == 5.0

    def test_zero_neurons_raises(self) -> None:
        """n_neurons=0 must raise ValueError."""
        with pytest.raises(ValueError):
            scpn_fusion_rs.PySnnPool(n_neurons=0, gain=1.0, window_size=10)

    def test_nan_gain_raises(self) -> None:
        """NaN gain must raise ValueError."""
        with pytest.raises(ValueError):
            scpn_fusion_rs.PySnnPool(n_neurons=10, gain=float("nan"), window_size=10)

    def test_zero_window_raises(self) -> None:
        """window_size=0 must raise ValueError."""
        with pytest.raises(ValueError):
            scpn_fusion_rs.PySnnPool(n_neurons=10, gain=1.0, window_size=0)

    def test_nan_error_raises(self) -> None:
        """NaN error input must raise ValueError."""
        pool = scpn_fusion_rs.PySnnPool(n_neurons=10, gain=1.0, window_size=10)
        with pytest.raises(ValueError):
            pool.step(float("nan"))

    def test_output_is_finite(self) -> None:
        """Output must be a finite float for valid inputs."""
        pool = scpn_fusion_rs.PySnnPool(n_neurons=20, gain=5.0, window_size=10)
        for _ in range(30):
            out = pool.step(1.0)
            assert math.isfinite(out), f"Non-finite output: {out}"


# ─── PySnnController tests ───


class TestPySnnController:
    def test_returns_control_signals(self) -> None:
        """Controller should return (ctrl_r, ctrl_z) tuple for position error."""
        ctrl = scpn_fusion_rs.PySnnController(target_r=6.2, target_z=0.0)
        # Drive with offset position for many steps to build up response
        ctrl_r = 0.0
        ctrl_z = 0.0
        for _ in range(50):
            ctrl_r, ctrl_z = ctrl.step(5.0, 1.0)
        # target_r=6.2, measured_r=5.0 => err_r=1.2 > 0 => ctrl_r > 0
        assert ctrl_r > 0.0, f"Expected positive ctrl_r, got {ctrl_r}"
        # target_z=0.0, measured_z=1.0 => err_z=-1.0 < 0 => ctrl_z < 0
        assert ctrl_z < 0.0, f"Expected negative ctrl_z, got {ctrl_z}"

    def test_getters(self) -> None:
        """Property getters should reflect constructor arguments."""
        ctrl = scpn_fusion_rs.PySnnController(target_r=6.5, target_z=-0.3)
        assert ctrl.target_r == 6.5
        assert ctrl.target_z == -0.3

    def test_nan_target_raises(self) -> None:
        """NaN target must raise ValueError."""
        with pytest.raises(ValueError):
            scpn_fusion_rs.PySnnController(target_r=float("nan"), target_z=0.0)

    def test_nan_measured_raises(self) -> None:
        """NaN measured position must raise ValueError."""
        ctrl = scpn_fusion_rs.PySnnController(target_r=6.2, target_z=0.0)
        with pytest.raises(ValueError):
            ctrl.step(float("nan"), 0.0)

    def test_inf_measured_raises(self) -> None:
        """Infinite measured position must raise ValueError."""
        ctrl = scpn_fusion_rs.PySnnController(target_r=6.2, target_z=0.0)
        with pytest.raises(ValueError):
            ctrl.step(6.1, float("inf"))


# ─── Python wrapper tests ───


class TestRustCompatWrappers:
    """Test the Python-side RustSnnPool / RustSnnController wrappers."""

    def test_rust_snn_pool_wrapper(self) -> None:
        from scpn_fusion.core._rust_compat import RustSnnPool

        pool = RustSnnPool(n_neurons=20, gain=5.0, window_size=10)
        assert pool.n_neurons == 20
        assert pool.gain == 5.0
        out = 0.0
        for _ in range(50):
            out = pool.step(3.0)
        assert out > 0.0

    def test_rust_snn_controller_wrapper(self) -> None:
        from scpn_fusion.core._rust_compat import RustSnnController

        ctrl = RustSnnController(target_r=6.2, target_z=0.0)
        assert ctrl.target_r == 6.2
        assert ctrl.target_z == 0.0
        cr = 0.0
        cz = 0.0
        for _ in range(50):
            cr, cz = ctrl.step(5.0, 1.0)
        assert cr > 0.0
        assert cz < 0.0

    def test_repr(self) -> None:
        from scpn_fusion.core._rust_compat import RustSnnPool, RustSnnController

        pool = RustSnnPool(n_neurons=30, gain=7.0)
        assert "30" in repr(pool)
        assert "7.0" in repr(pool)

        ctrl = RustSnnController(target_r=6.2, target_z=0.0)
        assert "6.2" in repr(ctrl)


# ─── Latency benchmark ───


class TestLatencyBenchmark:
    def test_rust_snn_throughput(self) -> None:
        """Measure wall-clock time for 10 000 Rust SNN steps."""
        pool = scpn_fusion_rs.PySnnPool(n_neurons=50, gain=10.0, window_size=20)
        n_steps = 10_000

        start = time.perf_counter()
        for i in range(n_steps):
            pool.step(0.5 * ((-1) ** i))
        elapsed_rust = time.perf_counter() - start

        # Sanity: 10k steps should complete in < 2 seconds (typically < 0.1s)
        assert elapsed_rust < 2.0, f"Rust SNN too slow: {elapsed_rust:.3f}s for {n_steps} steps"

    def test_rust_controller_throughput(self) -> None:
        """Measure wall-clock time for 10 000 Rust NeuroCyberneticController steps."""
        ctrl = scpn_fusion_rs.PySnnController(target_r=6.2, target_z=0.0)
        n_steps = 10_000

        start = time.perf_counter()
        for i in range(n_steps):
            ctrl.step(6.0 + 0.01 * i, 0.05 * ((-1) ** i))
        elapsed_rust = time.perf_counter() - start

        # Sanity: 10k steps should complete in < 5 seconds (typically < 0.2s)
        assert elapsed_rust < 5.0, f"Rust controller too slow: {elapsed_rust:.3f}s for {n_steps} steps"
