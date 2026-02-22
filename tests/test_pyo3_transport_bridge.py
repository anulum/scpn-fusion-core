# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — PyO3 Transport Bridge Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Tests for PyO3 bindings: Rust transport solver → Python.

Covers: WP-TR2 (Chang-Hinton chi, Sauter bootstrap, transport step).
"""

import time

import numpy as np
import pytest

try:
    import scpn_fusion_rs

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="scpn_fusion_rs not compiled")


def _iter_like_profiles(n=50):
    """Generate ITER-like radial profiles for testing."""
    rho = np.linspace(0.02, 1.0, n)
    t_i_kev = 15.0 * (1 - rho**2) ** 1.5 + 0.5
    t_e_kev = 12.0 * (1 - rho**2) ** 1.5 + 0.3
    n_e_19 = 10.0 * (1 - 0.7 * rho**2) + 1.0
    q = 1.0 + 2.5 * rho**2
    epsilon = rho * 0.32  # ITER: a/R ~ 0.32
    return rho, t_i_kev, t_e_kev, n_e_19, q, epsilon


class TestChangHintonProfile:
    """Tests for vectorized Chang-Hinton chi via Rust."""

    def test_returns_correct_length(self):
        rho, t_i, _, n_e, q, _ = _iter_like_profiles(50)
        solver = scpn_fusion_rs.PyTransportSolver()
        chi = solver.chang_hinton_chi_profile(rho, t_i, n_e, q)
        assert isinstance(chi, np.ndarray)
        assert len(chi) == 50

    def test_positive_values(self):
        rho, t_i, _, n_e, q, _ = _iter_like_profiles(50)
        solver = scpn_fusion_rs.PyTransportSolver()
        chi = solver.chang_hinton_chi_profile(rho, t_i, n_e, q)
        assert np.all(chi >= 0)
        assert np.all(np.isfinite(chi))

    def test_matches_python_implementation(self):
        """Rust profile must match Python scalar loop within 0.1% relative."""
        import scpn_fusion.core.integrated_transport_solver as its_mod
        from scpn_fusion.core.integrated_transport_solver import (
            IntegratedTransportSolver,
        )

        n = 50
        rho, t_i, _, n_e, q, _ = _iter_like_profiles(n)

        # Python path — disable Rust fast-path to exercise the scalar loop
        saved = its_mod._rust_transport_available
        its_mod._rust_transport_available = False
        try:
            solver_py = IntegratedTransportSolver.__new__(IntegratedTransportSolver)
            solver_py.rho = rho
            solver_py.t_i = t_i
            solver_py.n_e = n_e
            solver_py.q_profile = q
            chi_py = solver_py.chang_hinton_chi_profile()
        finally:
            its_mod._rust_transport_available = saved

        # Rust path
        solver_rs = scpn_fusion_rs.PyTransportSolver()
        chi_rs = solver_rs.chang_hinton_chi_profile(rho, t_i, n_e, q)

        np.testing.assert_allclose(chi_rs, chi_py, rtol=1e-3)


class TestSauterBootstrapProfile:
    """Tests for Sauter bootstrap current via Rust."""

    def test_returns_correct_length(self):
        rho, t_i, t_e, n_e, q, eps = _iter_like_profiles(50)
        solver = scpn_fusion_rs.PyTransportSolver()
        j_bs = solver.sauter_bootstrap_profile(rho, t_e, t_i, n_e, q, eps, 5.3)
        assert isinstance(j_bs, np.ndarray)
        assert len(j_bs) == 50

    def test_positive_bootstrap_current(self):
        rho, t_i, t_e, n_e, q, eps = _iter_like_profiles(50)
        solver = scpn_fusion_rs.PyTransportSolver()
        j_bs = solver.sauter_bootstrap_profile(rho, t_e, t_i, n_e, q, eps, 5.3)
        # Bootstrap current should be predominantly positive in tokamak
        assert np.mean(j_bs > 0) > 0.8


class TestTransportPerformance:
    """Benchmarks: Rust transport must be significantly faster than Python."""

    def test_rust_chi_faster_than_python(self):
        """Rust vectorized chi must be at least 5x faster than Python scalar loop."""
        import scpn_fusion.core.integrated_transport_solver as its_mod
        from scpn_fusion.core.integrated_transport_solver import (
            IntegratedTransportSolver,
        )

        n = 200
        rho, t_i, _, n_e, q, _ = _iter_like_profiles(n)
        n_reps = 100

        # Time Python — temporarily disable Rust fast-path so the
        # method exercises the actual Python for-loop.
        saved = its_mod._rust_transport_available
        its_mod._rust_transport_available = False
        try:
            solver_py = IntegratedTransportSolver.__new__(IntegratedTransportSolver)
            solver_py.rho = rho
            solver_py.t_i = t_i
            solver_py.n_e = n_e
            solver_py.q_profile = q
            t0 = time.perf_counter()
            for _ in range(n_reps):
                solver_py.chang_hinton_chi_profile()
            t_py = time.perf_counter() - t0
        finally:
            its_mod._rust_transport_available = saved

        # Time Rust
        solver_rs = scpn_fusion_rs.PyTransportSolver()
        t0 = time.perf_counter()
        for _ in range(n_reps):
            solver_rs.chang_hinton_chi_profile(rho, t_i, n_e, q)
        t_rs = time.perf_counter() - t0

        speedup = t_py / max(t_rs, 1e-9)
        print(f"Chang-Hinton chi speedup: {speedup:.1f}x (Python={t_py:.3f}s, Rust={t_rs:.3f}s)")
        assert speedup > 5.0, f"Expected >5x speedup, got {speedup:.1f}x"
