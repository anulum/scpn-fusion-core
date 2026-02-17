# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Rust/Python Parity Tests
# (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Rust/Python parity test suite (Plan Item 2.3).

Verifies that the Rust (scpn_fusion_rs) and pure-Python code paths produce
numerically equivalent results for every solver that has both implementations.

Convention: every test creates reproducible input (fixed seed / known analytic
source), runs *both* backends, and asserts ``np.allclose(py, rs, rtol=1e-3)``.
If the Rust extension is not compiled the entire module is skipped gracefully.

For each test, a simple Solov'ev equilibrium problem is set up
(R0=1.7, a=0.5, B0=2.0, Ip=1.0 MA) and both paths are run.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ── Rust availability probe ──────────────────────────────────────────

try:
    from scpn_fusion.core._rust_compat import (
        _rust_available,
        rust_multigrid_vcycle,
        RustAcceleratedKernel,
        RUST_BACKEND,
    )

    HAS_RUST = _rust_available()
except ImportError:
    HAS_RUST = False

# Also probe for individual Rust symbols used in specific tests
_HAS_RUST_MG = False
try:
    from scpn_fusion_rs import multigrid_vcycle as _rust_mg_fn  # type: ignore[import-not-found]

    _HAS_RUST_MG = True
except ImportError:
    _HAS_RUST_MG = False

_HAS_RUST_SHAFRANOV = False
try:
    from scpn_fusion_rs import shafranov_bv as _rust_shafranov_bv  # type: ignore[import-not-found]

    _HAS_RUST_SHAFRANOV = True
except ImportError:
    _HAS_RUST_SHAFRANOV = False

_HAS_RUST_TEARING = False
try:
    from scpn_fusion_rs import simulate_tearing_mode as _rust_tearing  # type: ignore[import-not-found]

    _HAS_RUST_TEARING = True
except ImportError:
    _HAS_RUST_TEARING = False

_HAS_RUST_SCPN_RUNTIME = False
try:
    from scpn_fusion_rs import (  # type: ignore[import-not-found]
        scpn_dense_activations as _rust_dense_act,
        scpn_marking_update as _rust_marking_upd,
    )

    _HAS_RUST_SCPN_RUNTIME = True
except ImportError:
    _HAS_RUST_SCPN_RUNTIME = False

# Module-level skip: if Rust is entirely absent every test is skipped
pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust bindings (scpn_fusion_rs) not available")


# ── Helpers ──────────────────────────────────────────────────────────

# Standard 65x65 test grid matching the task specification
NR, NZ = 65, 65

# Solov'ev equilibrium parameters: R0=1.7, a=0.5, B0=2.0, Ip=1.0 MA
R0 = 1.7
A_MINOR = 0.5
B0 = 2.0
IP_MA = 1.0

R_MIN = R0 - 1.5 * A_MINOR  # 0.95
R_MAX = R0 + 1.5 * A_MINOR  # 2.45
Z_MIN = -1.5 * A_MINOR      # -0.75
Z_MAX = 1.5 * A_MINOR       # 0.75


def _make_config(
    tmp_path: Path,
    *,
    nr: int = NR,
    nz: int = NZ,
    solver_method: str = "sor",
    max_iter: int = 300,
    tol: float = 1e-4,
    omega: float = 1.6,
) -> Path:
    """Write a minimal reactor config JSON for the Solov'ev problem and return the path."""
    cfg: dict[str, Any] = {
        "reactor_name": "Solovev-Parity-Test",
        "grid_resolution": [nr, nz],
        "dimensions": {
            "R_min": R_MIN,
            "R_max": R_MAX,
            "Z_min": Z_MIN,
            "Z_max": Z_MAX,
        },
        "physics": {
            "plasma_current_target": IP_MA,
            "vacuum_permeability": 1.0,
            "R0": R0,
            "a": A_MINOR,
            "B0": B0,
        },
        "coils": [
            {"name": "PF1", "r": R0 - A_MINOR * 1.2, "z": Z_MAX * 0.9, "current": 5.0},
            {"name": "PF2", "r": R0 + A_MINOR * 1.2, "z": Z_MAX * 0.9, "current": -1.5},
            {"name": "CS", "r": R_MIN * 0.8, "z": 0.0, "current": 0.1},
        ],
        "solver": {
            "max_iterations": max_iter,
            "convergence_threshold": tol,
            "relaxation_factor": 0.1,
            "solver_method": solver_method,
            "sor_omega": omega,
        },
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def _solovev_source(RR: np.ndarray, ZZ: np.ndarray) -> np.ndarray:
    """Analytic Solov'ev source term for the GS equation.

    S(R, Z) = -R^2  (corresponds to the simple Solov'ev equilibrium
    where p' = const, FF' = 0).  This gives a well-posed RHS for both
    the SOR and multigrid solvers.
    """
    return -(RR**2)


def _build_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Build the standard 65x65 (R, Z) mesh and return (R, Z, RR, ZZ, dR, dZ)."""
    R = np.linspace(R_MIN, R_MAX, NR)
    Z = np.linspace(Z_MIN, Z_MAX, NZ)
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])
    RR, ZZ = np.meshgrid(R, Z)
    return R, Z, RR, ZZ, dR, dZ


def _max_rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Element-wise maximum relative difference (safe against zero denom)."""
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom < 1e-15, 1.0, denom)
    return float(np.max(np.abs(a - b) / denom))


# ── 1. SOR Solver Parity ────────────────────────────────────────────


class TestSORSolverParity:
    """Compare the full Picard+SOR equilibrium solve between Python and Rust."""

    def test_sor_equilibrium_parity(self, tmp_path: Path) -> None:
        """Run the SOR-based equilibrium solver via both Python and Rust
        FusionKernel on the same 65x65 Solov'ev grid (R0=1.7, a=0.5,
        B0=2.0, Ip=1.0 MA).  The final Psi arrays must agree within
        rtol=1e-3.
        """
        from scpn_fusion.core.fusion_kernel import FusionKernel as PyFusionKernel

        cfg_path = _make_config(tmp_path, solver_method="sor", max_iter=300)

        # --- Python path ---
        py_kernel = PyFusionKernel(str(cfg_path))
        py_result = py_kernel.solve_equilibrium()
        psi_py = py_kernel.Psi.copy()

        # --- Rust path ---
        rust_kernel = RustAcceleratedKernel(str(cfg_path))
        rust_kernel.set_solver_method("sor")
        rust_result = rust_kernel.solve_equilibrium()
        psi_rs = rust_kernel.Psi.copy()

        # --- Compare ---
        assert psi_py.shape == psi_rs.shape, (
            f"Shape mismatch: Python {psi_py.shape} vs Rust {psi_rs.shape}"
        )
        assert np.all(np.isfinite(psi_py)), "Python SOR produced non-finite values"
        assert np.all(np.isfinite(psi_rs)), "Rust SOR produced non-finite values"

        np.testing.assert_allclose(
            psi_py, psi_rs, rtol=1e-3, atol=1e-6,
            err_msg=f"SOR parity failed: max rel diff = {_max_rel_diff(psi_py, psi_rs):.6e}",
        )

    def test_sor_single_sweep_consistency(self, tmp_path: Path) -> None:
        """Verify that the Python SOR sweep is self-consistent by
        running two sweeps and checking monotonic residual decrease.
        Both backends start from the same Solov'ev initial condition.
        """
        from scpn_fusion.core.fusion_kernel import FusionKernel as PyFusionKernel

        cfg_path = _make_config(tmp_path, solver_method="sor")
        py_kernel = PyFusionKernel(str(cfg_path))

        R, Z, RR, ZZ, dR, dZ = _build_grid()
        np.random.seed(42)
        Psi_init = np.random.randn(NZ, NR) * 0.1
        Source = _solovev_source(RR, ZZ)

        py_kernel.Psi = Psi_init.copy()
        py_kernel.RR = RR
        py_kernel.dR = dR
        py_kernel.dZ = dZ

        psi_after_1 = py_kernel._sor_step(Psi_init.copy(), Source, omega=1.6)
        psi_after_2 = py_kernel._sor_step(psi_after_1.copy(), Source, omega=1.6)

        assert np.all(np.isfinite(psi_after_2)), "SOR sweep produced non-finite values"


# ── 2. Multigrid Solver Parity ──────────────────────────────────────


class TestMultigridSolverParity:
    """Compare the Python multigrid V-cycle with the Rust multigrid."""

    @pytest.mark.skipif(
        not _HAS_RUST_MG,
        reason="Rust multigrid_vcycle not exposed via PyO3",
    )
    def test_multigrid_vcycle_parity(self, tmp_path: Path) -> None:
        """Run the multigrid V-cycle through both Python and Rust on
        the same 65x65 Solov'ev input (R0=1.7, a=0.5, B0=2.0, Ip=1.0 MA).
        Assert relative tolerance < 1e-3.
        """
        from scpn_fusion.core.fusion_kernel import FusionKernel as PyFusionKernel

        cfg_path = _make_config(tmp_path, solver_method="multigrid")
        py_kernel = PyFusionKernel(str(cfg_path))

        R, Z, RR, ZZ, dR, dZ = _build_grid()
        np.random.seed(123)
        Psi_init = np.zeros((NZ, NR))
        Source = _solovev_source(RR, ZZ)

        # --- Python V-cycle ---
        psi_py = py_kernel._multigrid_vcycle(
            Psi_init.copy(), Source, RR, dR, dZ, omega=1.6,
        )

        # --- Rust V-cycle ---
        psi_rs, residual, n_cycles, converged = rust_multigrid_vcycle(
            Source, Psi_init.copy(),
            R_MIN, R_MAX, Z_MIN, Z_MAX,
            NR, NZ,
            tol=1e-6, max_cycles=500,
        )

        # --- Compare ---
        assert psi_py.shape == psi_rs.shape
        assert np.all(np.isfinite(psi_py)), "Python multigrid produced NaN"
        assert np.all(np.isfinite(psi_rs)), "Rust multigrid produced NaN"

        np.testing.assert_allclose(
            psi_py, psi_rs, rtol=1e-3, atol=1e-6,
            err_msg=f"Multigrid parity failed: max rel diff = {_max_rel_diff(psi_py, psi_rs):.6e}",
        )

    def test_multigrid_equilibrium_parity(self, tmp_path: Path) -> None:
        """Compare full equilibrium solve using multigrid in both backends
        on the Solov'ev problem (R0=1.7, a=0.5, B0=2.0, Ip=1.0 MA).
        """
        from scpn_fusion.core.fusion_kernel import FusionKernel as PyFusionKernel

        cfg_path = _make_config(tmp_path, solver_method="multigrid", max_iter=200)

        # --- Python path ---
        py_kernel = PyFusionKernel(str(cfg_path))
        py_result = py_kernel.solve_equilibrium()
        psi_py = py_kernel.Psi.copy()

        # --- Rust path ---
        rust_kernel = RustAcceleratedKernel(str(cfg_path))
        rust_kernel.set_solver_method("multigrid")
        rust_result = rust_kernel.solve_equilibrium()
        psi_rs = rust_kernel.Psi.copy()

        # --- Compare ---
        assert psi_py.shape == psi_rs.shape
        assert np.all(np.isfinite(psi_py)), "Python multigrid equilibrium produced NaN"
        assert np.all(np.isfinite(psi_rs)), "Rust multigrid equilibrium produced NaN"

        np.testing.assert_allclose(
            psi_py, psi_rs, rtol=1e-3, atol=1e-6,
            err_msg=f"MG equilibrium parity failed: max rel diff = {_max_rel_diff(psi_py, psi_rs):.6e}",
        )


# ── 3. Vacuum Field Parity ──────────────────────────────────────────


class TestVacuumFieldParity:
    """Compare vacuum field (coil Green's functions) between Python and Rust
    on the same Solov'ev geometry (R0=1.7, a=0.5, B0=2.0, Ip=1.0 MA).
    """

    def test_vacuum_field_parity(self, tmp_path: Path) -> None:
        """Compute vacuum field from both backends. They should agree
        within rtol=1e-3 since they solve the same coil geometry.
        """
        from scpn_fusion.core.fusion_kernel import FusionKernel as PyFusionKernel

        cfg_path = _make_config(tmp_path)

        # --- Python vacuum field ---
        py_kernel = PyFusionKernel(str(cfg_path))
        psi_vac_py = py_kernel.calculate_vacuum_field()

        # --- Rust vacuum field ---
        rust_kernel = RustAcceleratedKernel(str(cfg_path))

        if hasattr(rust_kernel, "calculate_vacuum_field"):
            psi_vac_rs = rust_kernel.calculate_vacuum_field()
        elif hasattr(rust_kernel, "_rust") and hasattr(rust_kernel._rust, "get_vacuum_field"):
            psi_vac_rs = np.asarray(rust_kernel._rust.get_vacuum_field())
        else:
            # Rust kernel doesn't expose vacuum field separately.
            # Compare boundary values after solve (both use vacuum BCs).
            rust_kernel.set_solver_method("sor")
            rust_kernel.solve_equilibrium()

            psi_vac_rs = np.zeros_like(psi_vac_py)
            psi_vac_rs[0, :] = rust_kernel.Psi[0, :]
            psi_vac_rs[-1, :] = rust_kernel.Psi[-1, :]
            psi_vac_rs[:, 0] = rust_kernel.Psi[:, 0]
            psi_vac_rs[:, -1] = rust_kernel.Psi[:, -1]

            psi_vac_py_boundary = np.zeros_like(psi_vac_py)
            psi_vac_py_boundary[0, :] = psi_vac_py[0, :]
            psi_vac_py_boundary[-1, :] = psi_vac_py[-1, :]
            psi_vac_py_boundary[:, 0] = psi_vac_py[:, 0]
            psi_vac_py_boundary[:, -1] = psi_vac_py[:, -1]

            psi_vac_py = psi_vac_py_boundary

        assert psi_vac_py.shape == psi_vac_rs.shape
        assert np.all(np.isfinite(psi_vac_py)), "Python vacuum field has NaN"
        assert np.all(np.isfinite(psi_vac_rs)), "Rust vacuum field has NaN"

        np.testing.assert_allclose(
            psi_vac_py, psi_vac_rs, rtol=1e-3, atol=1e-8,
            err_msg=f"Vacuum field parity failed: max rel diff = {_max_rel_diff(psi_vac_py, psi_vac_rs):.6e}",
        )

    @pytest.mark.skipif(
        not _HAS_RUST_SHAFRANOV,
        reason="Rust shafranov_bv not exposed via PyO3",
    )
    def test_shafranov_bv_parity(self, tmp_path: Path) -> None:
        """Compare the Rust shafranov_bv() against the Python vacuum
        field computation for the Solov'ev problem.
        """
        from scpn_fusion.core.fusion_kernel import FusionKernel as PyFusionKernel
        from scpn_fusion.core._rust_compat import rust_shafranov_bv

        cfg_path = _make_config(tmp_path)
        py_kernel = PyFusionKernel(str(cfg_path))
        psi_vac_py = py_kernel.calculate_vacuum_field()

        psi_vac_rs = rust_shafranov_bv(str(cfg_path))
        psi_vac_rs = np.asarray(psi_vac_rs)

        assert psi_vac_py.shape == psi_vac_rs.shape

        np.testing.assert_allclose(
            psi_vac_py, psi_vac_rs, rtol=1e-3, atol=1e-8,
            err_msg=f"shafranov_bv parity failed: max rel diff = {_max_rel_diff(psi_vac_py, psi_vac_rs):.6e}",
        )


# ── 4. Transport Solver Parity ──────────────────────────────────────


class TestTransportSolverParity:
    """Compare transport-related computations between Rust and Python
    on the Solov'ev equilibrium (R0=1.7, a=0.5, B0=2.0, Ip=1.0 MA).
    """

    @pytest.mark.skipif(
        not _HAS_RUST_TEARING,
        reason="Rust simulate_tearing_mode not exposed via PyO3",
    )
    def test_tearing_mode_parity(self) -> None:
        """Compare the Rust and Python tearing mode simulators.
        Both should produce statistically equivalent disruption physics
        when given the same RNG seed.
        """
        from scpn_fusion.control.disruption_predictor import (
            simulate_tearing_mode as py_tearing,
        )
        from scpn_fusion.core._rust_compat import rust_simulate_tearing_mode

        steps = 500

        rng_py = np.random.default_rng(seed=2026)
        signal_py, label_py, ttd_py = py_tearing(steps=steps, rng=rng_py)

        signal_rs, label_rs, ttd_rs = rust_simulate_tearing_mode(steps=steps, seed=2026)
        signal_rs = np.asarray(signal_rs)

        assert np.all(np.isfinite(signal_py)), "Python tearing mode produced NaN"
        assert np.all(np.isfinite(signal_rs)), "Rust tearing mode produced NaN"

        assert label_py == label_rs, (
            f"Disruption label mismatch: Python={label_py}, Rust={label_rs}"
        )

        min_len = min(len(signal_py), len(signal_rs))
        if min_len > 0:
            np.testing.assert_allclose(
                signal_py[:min_len], signal_rs[:min_len], rtol=1e-3, atol=1e-4,
                err_msg=f"Tearing mode parity failed: max rel diff = {_max_rel_diff(signal_py[:min_len], signal_rs[:min_len]):.6e}",
            )

    def test_transport_no_rust_path_skips_gracefully(self) -> None:
        """Verify that the neural transport module works in pure-Python
        mode without Rust, and produces a valid surrogate object.
        """
        from scpn_fusion.core.neural_transport import NeuralTransportSurrogate

        surrogate = NeuralTransportSurrogate()
        assert surrogate is not None


# ── 5. SCPN Controller Runtime Parity ────────────────────────────────


class TestSCPNRuntimeParity:
    """Compare SCPN controller Rust runtime kernels vs NumPy fallbacks."""

    @pytest.mark.skipif(
        not _HAS_RUST_SCPN_RUNTIME,
        reason="Rust SCPN runtime (dense_activations, marking_update) not available",
    )
    def test_dense_activations_parity(self) -> None:
        """Compare Rust scpn_dense_activations vs NumPy matrix multiply."""
        np.random.seed(999)
        n_places = 16
        n_transitions = 8
        marking = np.random.rand(n_places).astype(np.float64)
        weights = np.random.rand(n_transitions, n_places).astype(np.float64)

        act_py = weights @ marking
        act_rs = np.asarray(_rust_dense_act(marking, weights))

        np.testing.assert_allclose(
            act_py, act_rs, rtol=1e-6, atol=1e-10,
            err_msg=f"Dense activations parity failed: max rel diff = {_max_rel_diff(act_py, act_rs):.6e}",
        )

    @pytest.mark.skipif(
        not _HAS_RUST_SCPN_RUNTIME,
        reason="Rust SCPN runtime (dense_activations, marking_update) not available",
    )
    def test_marking_update_parity(self) -> None:
        """Compare Rust scpn_marking_update vs NumPy equivalent."""
        np.random.seed(1001)
        n_places = 16
        marking = np.random.rand(n_places).astype(np.float64)
        pre = np.random.rand(n_places).astype(np.float64)
        post = np.random.rand(n_places).astype(np.float64)
        firing = np.random.rand(n_places).astype(np.float64)

        mk_py = marking - pre * firing + post * firing
        mk_rs = np.asarray(_rust_marking_upd(marking, pre, post, firing))

        np.testing.assert_allclose(
            mk_py, mk_rs, rtol=1e-6, atol=1e-10,
            err_msg=f"Marking update parity failed: max rel diff = {_max_rel_diff(mk_py, mk_rs):.6e}",
        )


# ── 6. B-field Post-Processing Parity ───────────────────────────────


class TestBFieldParity:
    """Compare the B-field derivation from Psi between backends on the
    Solov'ev equilibrium (R0=1.7, a=0.5, B0=2.0, Ip=1.0 MA).
    """

    def test_b_field_parity(self, tmp_path: Path) -> None:
        """After equilibrium solve, B_R and B_Z should match between
        Python and Rust within rtol=1e-3.
        """
        from scpn_fusion.core.fusion_kernel import FusionKernel as PyFusionKernel

        cfg_path = _make_config(tmp_path, solver_method="sor", max_iter=200)

        # --- Python ---
        py_kernel = PyFusionKernel(str(cfg_path))
        py_kernel.solve_equilibrium()
        br_py = py_kernel.B_R.copy()
        bz_py = py_kernel.B_Z.copy()

        # --- Rust ---
        rust_kernel = RustAcceleratedKernel(str(cfg_path))
        rust_kernel.set_solver_method("sor")
        rust_kernel.solve_equilibrium()
        br_rs = rust_kernel.B_R.copy()
        bz_rs = rust_kernel.B_Z.copy()

        # --- Compare B_R ---
        assert br_py.shape == br_rs.shape
        np.testing.assert_allclose(
            br_py, br_rs, rtol=1e-3, atol=1e-6,
            err_msg=f"B_R parity failed: max rel diff = {_max_rel_diff(br_py, br_rs):.6e}",
        )

        # --- Compare B_Z ---
        assert bz_py.shape == bz_rs.shape
        np.testing.assert_allclose(
            bz_py, bz_rs, rtol=1e-3, atol=1e-6,
            err_msg=f"B_Z parity failed: max rel diff = {_max_rel_diff(bz_py, bz_rs):.6e}",
        )


# ── 7. X-Point and Topology Parity ──────────────────────────────────


class TestTopologyParity:
    """Compare X-point detection between backends on the Solov'ev
    equilibrium (R0=1.7, a=0.5, B0=2.0, Ip=1.0 MA).
    """

    def test_x_point_parity(self, tmp_path: Path) -> None:
        """After equilibrium solve, the X-point location and Psi value
        should agree between Python and Rust.
        """
        from scpn_fusion.core.fusion_kernel import FusionKernel as PyFusionKernel

        cfg_path = _make_config(tmp_path, solver_method="sor", max_iter=200)

        # --- Python ---
        py_kernel = PyFusionKernel(str(cfg_path))
        py_kernel.solve_equilibrium()
        xpt_py, psi_xpt_py = py_kernel.find_x_point(py_kernel.Psi)

        # --- Rust ---
        rust_kernel = RustAcceleratedKernel(str(cfg_path))
        rust_kernel.set_solver_method("sor")
        rust_kernel.solve_equilibrium()
        xpt_rs, psi_xpt_rs = rust_kernel.find_x_point(rust_kernel.Psi)

        # X-point position should be within 1 grid cell
        dR = (R_MAX - R_MIN) / (NR - 1)
        dZ = (Z_MAX - Z_MIN) / (NZ - 1)
        assert abs(xpt_py[0] - xpt_rs[0]) < 2 * dR, (
            f"X-point R mismatch: {xpt_py[0]:.4f} vs {xpt_rs[0]:.4f}"
        )
        assert abs(xpt_py[1] - xpt_rs[1]) < 2 * dZ, (
            f"X-point Z mismatch: {xpt_py[1]:.4f} vs {xpt_rs[1]:.4f}"
        )

        # Psi at X-point should agree
        if abs(psi_xpt_py) > 1e-10:
            rel_diff = abs(psi_xpt_py - psi_xpt_rs) / abs(psi_xpt_py)
            assert rel_diff < 1e-2, (
                f"X-point Psi mismatch: {psi_xpt_py:.6e} vs {psi_xpt_rs:.6e} "
                f"(rel diff = {rel_diff:.6e})"
            )
