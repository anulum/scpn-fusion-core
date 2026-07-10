# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Newton Equilibrium Dispatch Tests
"""Dispatch and fault-path tests for the FusionKernel Newton solver mixin.

Cover the method-dispatch fan-out (``rust_multigrid`` / ``newton`` / ``anderson``
/ under-relaxation), the zero-current vacuum short circuit, the
Newton-with-line-search path, and the divergence/validation fail-closed
branches. The physics-convergence behaviour is exercised end to end by
``test_newton_equilibrium``; this module instead drives the branch structure of
the dispatcher, injecting degenerate topology, zero-length iteration budgets,
and linear-solve faults by monkeypatching the kernel's elliptic solve, residual
helpers and Newton linear-system helper, since a well-posed screening
equilibrium does not reach those guards on its own.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_fusion.core import fusion_kernel_newton_solver as newton_module
from scpn_fusion.core.fusion_kernel import FusionKernel

_BASE_CONFIG: dict[str, Any] = {
    "reactor_name": "Newton-Solver-Test",
    "grid_resolution": [16, 16],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [{"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15}],
    "solver": {
        "max_iterations": 30,
        "convergence_threshold": 1e-4,
        "relaxation_factor": 0.1,
        "solver_method": "newton",
    },
}


def _kernel(tmp_path: Path, **solver_overrides: Any) -> FusionKernel:
    """Build a FusionKernel with solver-config overrides applied."""
    cfg = deepcopy(_BASE_CONFIG)
    cfg["solver"].update(solver_overrides)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return FusionKernel(str(path))


def _nan_like(kernel: FusionKernel) -> Any:
    """Return a NaN-filled array shaped like the kernel flux grid."""
    return np.full_like(kernel.Psi, np.nan)


def _pin_close_topology(kernel: FusionKernel, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force axis and boundary flux within 0.1 so the degenerate-gap guard fires."""
    monkeypatch.setattr(kernel, "_find_magnetic_axis", lambda: (0, 0, 1.0))
    monkeypatch.setattr(kernel, "find_x_point", lambda Psi: ((0.0, 0.0), 1.02))


class TestMethodDispatch:
    """Route ``solve_equilibrium`` through each supported solver method."""

    def test_newton_method_runs(self, tmp_path: Path) -> None:
        """The Newton dispatch returns a Newton-labelled result dict."""
        result = _kernel(tmp_path, solver_method="newton").solve_equilibrium()
        assert result["solver_method"] == "newton"
        assert np.isfinite(result["psi"]).all()

    def test_newton_line_search_runs(self, tmp_path: Path) -> None:
        """Newton with Armijo line search populates line-search telemetry."""
        result = _kernel(
            tmp_path,
            solver_method="newton",
            newton_line_search=True,
            newton_line_search_c=1.0e-4,
            newton_line_search_max_backtracks=4,
        ).solve_equilibrium()
        assert result["newton_line_search"] is True
        assert result["newton_line_search_attempts"] >= 1

    def test_anderson_method_runs(self, tmp_path: Path) -> None:
        """The Anderson-accelerated under-relaxation path runs to completion."""
        result = _kernel(
            tmp_path,
            solver_method="anderson",
            max_iterations=12,
            anderson_depth=5,
        ).solve_equilibrium()
        assert result["solver_method"] == "anderson"

    def test_rust_multigrid_method_runs(self, tmp_path: Path) -> None:
        """The rust_multigrid fast path delegates and returns a result dict."""
        result = _kernel(tmp_path, solver_method="rust_multigrid").solve_equilibrium()
        assert result["solver_method"] == "rust_multigrid"
        assert "psi" in result

    def test_zero_current_vacuum_short_circuit(self, tmp_path: Path) -> None:
        """A zero plasma-current target returns the vacuum field in zero iters."""
        cfg = deepcopy(_BASE_CONFIG)
        cfg["solver"]["solver_method"] = "sor"
        cfg["physics"]["plasma_current_target"] = 0.0
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        result = FusionKernel(str(path)).solve_equilibrium()
        assert result["converged"] is True
        assert result["iterations"] == 0
        assert result["residual"] == 0.0


class TestConfigValidation:
    """Reject inconsistent solver configuration at solve time."""

    def test_gs_residual_threshold_must_be_positive_newton(self, tmp_path: Path) -> None:
        """Newton dispatch rejects a non-positive GS residual threshold."""
        kernel = _kernel(
            tmp_path,
            solver_method="newton",
            require_gs_residual=True,
            gs_residual_threshold=0.0,
        )
        with pytest.raises(ValueError, match="gs_residual_threshold must be > 0"):
            kernel.solve_equilibrium()

    def test_gs_residual_threshold_must_be_positive_relaxation(self, tmp_path: Path) -> None:
        """Under-relaxation dispatch rejects a non-positive GS residual threshold."""
        kernel = _kernel(
            tmp_path,
            solver_method="sor",
            require_gs_residual=True,
            gs_residual_threshold=0.0,
        )
        with pytest.raises(ValueError, match="gs_residual_threshold must be > 0"):
            kernel.solve_equilibrium()


class TestDivergenceHandling:
    """Cover the NaN-divergence branches of both solver families."""

    def test_relaxation_divergence_reverts_and_breaks(self, tmp_path: Path) -> None:
        """A diverging under-relaxation solve reverts to best state and stops."""
        kernel = _kernel(tmp_path, solver_method="sor")
        kernel._elliptic_solve = lambda *a, **k: _nan_like(kernel)  # type: ignore[method-assign]  # forced divergence
        result = kernel.solve_equilibrium()
        assert result["converged"] is False

    def test_relaxation_divergence_fails_closed(self, tmp_path: Path) -> None:
        """A diverging under-relaxation solve raises when fail_on_diverge is set."""
        kernel = _kernel(tmp_path, solver_method="sor", fail_on_diverge=True)
        kernel._elliptic_solve = lambda *a, **k: _nan_like(kernel)  # type: ignore[method-assign]  # forced divergence
        with pytest.raises(RuntimeError, match="diverged"):
            kernel.solve_equilibrium()

    def test_newton_warmup_divergence_breaks(self, tmp_path: Path) -> None:
        """A diverging Newton Picard warmup breaks out without raising."""
        kernel = _kernel(tmp_path, solver_method="newton")
        kernel._elliptic_solve = lambda *a, **k: _nan_like(kernel)  # type: ignore[method-assign]  # forced divergence
        result = kernel.solve_equilibrium()
        assert result["solver_method"] == "newton"

    def test_newton_warmup_divergence_fails_closed(self, tmp_path: Path) -> None:
        """A diverging Newton warmup raises when fail_on_diverge is set."""
        kernel = _kernel(tmp_path, solver_method="newton", fail_on_diverge=True)
        kernel._elliptic_solve = lambda *a, **k: _nan_like(kernel)  # type: ignore[method-assign]  # forced divergence
        with pytest.raises(RuntimeError, match="diverged"):
            kernel.solve_equilibrium()


class TestEmptyIterationBudget:
    """A zero-length iteration budget leaves no source and reports infinite residual.

    The config schema forbids ``max_iterations == 0`` at construction, so the
    zero-budget guard is induced by mutating the live runtime ``cfg`` dict that
    both dispatchers read — the branch it defends is otherwise unreachable.
    """

    def test_newton_zero_iterations_reports_infinite_residual(self, tmp_path: Path) -> None:
        """Newton with no iterations never builds a source (final_source is None)."""
        kernel = _kernel(tmp_path, solver_method="newton")
        kernel.cfg["solver"]["max_iterations"] = 0
        result = kernel.solve_equilibrium()
        assert result["gs_residual"] == float("inf")
        assert result["gs_residual_best"] == float("inf")
        assert result["converged"] is False

    def test_relaxation_zero_iterations_reports_infinite_residual(self, tmp_path: Path) -> None:
        """Under-relaxation with no iterations never builds a source either."""
        kernel = _kernel(tmp_path, solver_method="sor")
        kernel.cfg["solver"]["max_iterations"] = 0
        result = kernel.solve_equilibrium()
        assert result["gs_residual"] == float("inf")
        assert result["gs_residual_best"] == float("inf")
        assert result["converged"] is False


class TestDegenerateTopologyConvergence:
    """Convergence branches reached under a pinned, degenerate flux topology."""

    def test_newton_warmup_converges_on_identity_elliptic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An identity elliptic step drives the Picard warmup to converge immediately."""
        kernel = _kernel(tmp_path, solver_method="newton", max_iterations=4)
        _pin_close_topology(kernel, monkeypatch)
        monkeypatch.setattr(kernel, "_elliptic_solve", lambda Source, bc: kernel.Psi.copy())
        result = kernel.solve_equilibrium()
        assert result["converged"] is True
        assert result["solver_method"] == "newton"

    def test_relaxation_converges_with_close_axis(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A degenerate axis/boundary gap plus identity elliptic converges in one step."""
        kernel = _kernel(tmp_path, solver_method="sor", max_iterations=5)
        _pin_close_topology(kernel, monkeypatch)
        monkeypatch.setattr(kernel, "_elliptic_solve", lambda Source, bc: kernel.Psi.copy())
        result = kernel.solve_equilibrium()
        assert result["converged"] is True
        assert result["iterations"] == 1


class TestNewtonPhaseBBranches:
    """Newton second-phase convergence, rejection and divergence branches."""

    def test_phase_b_converges_with_line_search(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A zero GS residual in phase B converges under the line-search path."""
        kernel = _kernel(
            tmp_path,
            solver_method="newton",
            max_iterations=2,
            newton_line_search=True,
            newton_line_search_c=1e-4,
        )
        # Warmup uses the RMS helper; force it non-convergent, then zero the phase-B
        # residual so the Newton loop converges on its first step.
        monkeypatch.setattr(kernel, "_elliptic_solve", lambda Source, bc: kernel.Psi + 1.0)
        monkeypatch.setattr(
            kernel, "_compute_gs_residual", lambda Source: np.zeros_like(kernel.Psi)
        )
        result = kernel.solve_equilibrium()
        assert result["converged"] is True
        assert result["newton_line_search"] is True

    def test_phase_b_line_search_rejects_all_backtracks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-decreasing residual rejects every backtrack and falls back to the min step."""
        kernel = _kernel(
            tmp_path,
            solver_method="newton",
            max_iterations=2,
            newton_line_search=True,
            newton_line_search_c=1e-4,
            newton_line_search_max_backtracks=2,
        )
        _pin_close_topology(kernel, monkeypatch)
        monkeypatch.setattr(kernel, "_elliptic_solve", lambda Source, bc: kernel.Psi + 1.0)
        monkeypatch.setattr(
            kernel, "_compute_gs_residual", lambda Source: np.full_like(kernel.Psi, 5.0)
        )
        monkeypatch.setattr(
            newton_module,
            "_solve_newton_linear_system_runtime",
            lambda **kw: (np.zeros(kw["n_interior"]), 0),
        )
        result = kernel.solve_equilibrium()
        assert result["newton_line_search_rejects"] >= 1
        assert result["converged"] is False

    def test_phase_b_divergence_breaks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A NaN Newton step breaks out of phase B without raising by default."""
        kernel = _kernel(tmp_path, solver_method="newton", max_iterations=2)
        monkeypatch.setattr(kernel, "_elliptic_solve", lambda Source, bc: kernel.Psi + 1.0)
        monkeypatch.setattr(
            kernel, "_compute_gs_residual", lambda Source: np.full_like(kernel.Psi, 1.0)
        )
        monkeypatch.setattr(
            newton_module,
            "_solve_newton_linear_system_runtime",
            lambda **kw: (np.full(kw["n_interior"], np.nan), 0),
        )
        result = kernel.solve_equilibrium()
        assert result["solver_method"] == "newton"

    def test_phase_b_divergence_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A NaN Newton step raises in phase B when fail_on_diverge is set."""
        kernel = _kernel(tmp_path, solver_method="newton", max_iterations=2, fail_on_diverge=True)
        monkeypatch.setattr(kernel, "_elliptic_solve", lambda Source, bc: kernel.Psi + 1.0)
        monkeypatch.setattr(
            kernel, "_compute_gs_residual", lambda Source: np.full_like(kernel.Psi, 1.0)
        )
        monkeypatch.setattr(
            newton_module,
            "_solve_newton_linear_system_runtime",
            lambda **kw: (np.full(kw["n_interior"], np.nan), 0),
        )
        with pytest.raises(RuntimeError, match="diverged"):
            kernel.solve_equilibrium()
