# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — PyO3 GPU Bridge Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Tests for PyO3 bindings: fusion-gpu crate → Python.

Covers: WP-GPU1 (GpuGsSolver, gpu_available, gpu_info).

These tests require the Rust extension built with --features gpu:
    cd scpn-fusion-rs/crates/fusion-python && maturin develop --release --features gpu
"""

import numpy as np
import pytest

try:
    import scpn_fusion_rs

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

# GPU functions may not exist if built without `gpu` feature
HAS_GPU_BINDINGS = HAS_RUST and hasattr(scpn_fusion_rs, "py_gpu_available")
HAS_GPU_DEVICE = HAS_GPU_BINDINGS and scpn_fusion_rs.py_gpu_available()

pytestmark = pytest.mark.skipif(not HAS_GPU_BINDINGS, reason="GPU bindings not compiled")


class TestGpuAvailability:
    """Test GPU availability detection functions."""

    def test_gpu_available_returns_bool(self):
        result = scpn_fusion_rs.py_gpu_available()
        assert isinstance(result, bool)

    def test_gpu_info_returns_string_or_none(self):
        result = scpn_fusion_rs.py_gpu_info()
        assert result is None or isinstance(result, str)


@pytest.mark.skipif(not HAS_GPU_DEVICE, reason="No GPU device available")
class TestPyGpuSolver:
    """Tests for GpuGsSolver binding. Only run when GPU hardware is present."""

    def test_construction(self):
        solver = scpn_fusion_rs.PyGpuSolver(65, 65, 1.0, 9.0, -5.0, 5.0)
        assert solver.grid_shape() == (65, 65)

    def test_solve_returns_finite(self):
        nr, nz = 33, 33
        solver = scpn_fusion_rs.PyGpuSolver(nr, nz, 1.0, 9.0, -5.0, 5.0)
        psi = np.zeros(nr * nz, dtype=np.float32)
        source = np.full(nr * nz, -1.0, dtype=np.float32)
        result = solver.solve(psi.tolist(), source.tolist(), 100, 1.5)
        assert isinstance(result, np.ndarray)
        assert len(result) == nr * nz
        assert np.all(np.isfinite(result))

    def test_solve_nonzero(self):
        """Non-zero source should produce non-zero solution."""
        nr, nz = 33, 33
        solver = scpn_fusion_rs.PyGpuSolver(nr, nz, 1.0, 9.0, -5.0, 5.0)
        psi = np.zeros(nr * nz, dtype=np.float32)
        source = np.full(nr * nz, -1.0, dtype=np.float32)
        result = solver.solve(psi.tolist(), source.tolist(), 200, 1.5)
        assert np.max(np.abs(result)) > 0.0
