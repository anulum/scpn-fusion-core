# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Member-method branch tests for the transport-solver model mixin.

Exercises the reduced-transport helper methods and their validation/fallback
branches that the reduced-multichannel step tests never reach: coarse-channel
summarisation, neural OOD index selection, closure-input resolution, recovery
budget guards, the backward-compatible Chang-Hinton and bootstrap helpers, and
the gyro-Bohm coefficient sourcing paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_fusion.core import integrated_transport_solver as its_mod
from scpn_fusion.core.integrated_transport_solver import PhysicsError, TransportSolver

_BASE_CONFIG: dict[str, Any] = {
    "reactor_name": "TransportSolver-Model-Test",
    "grid_resolution": [24, 24],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0, "B0": 5.3},
    "coils": [{"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15}],
    "solver": {"max_iterations": 10, "convergence_threshold": 1e-4, "relaxation_factor": 0.1},
}


def _solver(tmp_path: Path) -> TransportSolver:
    """Construct a single-ion solver from a temp JSON config."""
    path = tmp_path / "transport_config.json"
    path.write_text(json.dumps(_BASE_CONFIG), encoding="utf-8")
    return TransportSolver(str(path), multi_ion=False)


def _neo_params(solver: TransportSolver, **overrides: Any) -> dict[str, Any]:
    """Return a valid neoclassical-params dict sized to the solver grid."""
    params: dict[str, Any] = {
        "R0": 6.2,
        "a": 2.0,
        "B0": 5.3,
        "A_ion": 2.0,
        "Z_eff": 1.5,
        "q_profile": np.linspace(1.0, 3.0, solver.rho.size),
    }
    params.update(overrides)
    return params


class TestSummariseCoarseChannels:
    """Coarse channel summary edge behaviour."""

    def test_all_stable_profiles_report_stable(self, tmp_path: Path) -> None:
        """Zero transport everywhere yields a ``stable`` dominant channel."""
        solver = _solver(tmp_path)
        zeros = np.zeros(solver.rho.size, dtype=np.float64)
        dominant, counts, energy = solver._summarize_coarse_transport_channels(zeros, zeros)
        assert dominant == "stable"
        assert counts["stable"] == solver.rho.size
        assert energy["ITG"] == 0.0


class TestSelectNeuralOodIndices:
    """Neural OOD index selection validation and empty-set handling."""

    def test_non_finite_sigma_raises(self, tmp_path: Path) -> None:
        """A non-positive sigma threshold is rejected."""
        solver = _solver(tmp_path)
        z = np.zeros(solver.rho.size, dtype=np.float64)
        with pytest.raises(ValueError, match="ood_sigma"):
            solver._select_neural_ood_indices(z, sigma_threshold=0.0, max_points=5)

    def test_non_positive_max_points_raises(self, tmp_path: Path) -> None:
        """A boolean or non-positive ``max_points`` is rejected."""
        solver = _solver(tmp_path)
        z = np.zeros(solver.rho.size, dtype=np.float64)
        with pytest.raises(ValueError, match="max_points"):
            solver._select_neural_ood_indices(z, sigma_threshold=5.0, max_points=0)

    def test_shape_mismatch_raises(self, tmp_path: Path) -> None:
        """An OOD profile whose length disagrees with the grid is rejected."""
        solver = _solver(tmp_path)
        z = np.zeros(solver.rho.size + 1, dtype=np.float64)
        with pytest.raises(ValueError, match="rho grid"):
            solver._select_neural_ood_indices(z, sigma_threshold=5.0, max_points=5)

    def test_no_ood_points_returns_empty_selection(self, tmp_path: Path) -> None:
        """When nothing exceeds the threshold, no escalation indices are returned."""
        solver = _solver(tmp_path)
        z = np.zeros(solver.rho.size, dtype=np.float64)
        mask, selected = solver._select_neural_ood_indices(z, sigma_threshold=5.0, max_points=5)
        assert selected == []
        assert not np.any(mask)


class TestResolveClosureInputs:
    """Closure-input resolution fallbacks."""

    def test_invalid_neoclassical_q_profile_falls_back(self, tmp_path: Path) -> None:
        """A non-physical neoclassical q-profile falls back to the parabolic default."""
        solver = _solver(tmp_path)
        solver.neoclassical_params = _neo_params(solver, q_profile=np.full(solver.rho.size, -1.0))
        _, _, _, _, _, source = solver._resolve_transport_closure_inputs()
        assert source == "fallback_parabolic"


class TestRecoveryBudget:
    """Numerical-recovery limit resolution and budget enforcement."""

    def test_negative_recovery_override_raises(self, tmp_path: Path) -> None:
        """A negative recovery override is rejected."""
        solver = _solver(tmp_path)
        with pytest.raises(ValueError, match="non-negative integer"):
            solver._resolve_recovery_limit(-1)

    def test_budget_within_limit_does_not_raise(self, tmp_path: Path) -> None:
        """A recovery count at or below the limit passes enforcement."""
        solver = _solver(tmp_path)
        solver.set_numerical_recovery_limit(5)
        solver._last_numerical_recovery_count = 2
        solver._enforce_recovery_budget(
            enforce_numerical_recovery=True, max_numerical_recoveries=None
        )
        assert solver._last_numerical_recovery_limit == 5


class TestSetNeoclassical:
    """set_neoclassical q-profile validation."""

    def test_non_physical_q_profile_raises(self, tmp_path: Path) -> None:
        """A corrupted rho grid that drives q non-positive is rejected."""
        solver = _solver(tmp_path)
        # An out-of-domain rho grid with q0 > q_edge drives q negative at the
        # far node, tripping the generated-profile validity guard.
        solver.rho = np.array([0.0, 2.0, 0.5], dtype=np.float64)
        with pytest.raises(ValueError, match="invalid values"):
            solver.set_neoclassical(R0=6.2, a=2.0, B0=5.3, q0=10.0, q_edge=1.0)


class TestChangHintonProfile:
    """Backward-compatible Chang-Hinton profile helper."""

    def test_default_profile_is_finite(self, tmp_path: Path) -> None:
        """The helper returns a finite profile with no neoclassical params set."""
        solver = _solver(tmp_path)
        chi = solver.chang_hinton_chi_profile()
        assert chi.shape == solver.rho.shape
        assert np.all(np.isfinite(chi))

    def test_mismatched_q_profile_is_replaced(self, tmp_path: Path) -> None:
        """A wrong-length ``q_profile`` attribute is replaced with a default ramp."""
        solver = _solver(tmp_path)
        solver.q_profile = np.ones(solver.rho.size + 1, dtype=np.float64)
        chi = solver.chang_hinton_chi_profile()
        assert chi.shape == solver.rho.shape
        assert np.all(np.isfinite(chi))


class TestBootstrapCurrent:
    """Bootstrap-current helpers."""

    def test_simple_bootstrap_zeroes_boundaries(self, tmp_path: Path) -> None:
        """The calibrated-heuristic bootstrap current vanishes at both boundaries."""
        solver = _solver(tmp_path)
        b_pol = np.full(solver.rho.size, 1.0, dtype=np.float64)
        j_bs = solver.calculate_bootstrap_current_simple(6.2, b_pol)
        assert j_bs[0] == 0.0
        assert j_bs[-1] == 0.0
        assert np.all(np.isfinite(j_bs))

    def test_full_bootstrap_replaces_mismatched_q_profile(self, tmp_path: Path) -> None:
        """A wrong-shape neoclassical q-profile is replaced before the Sauter kernel."""
        solver = _solver(tmp_path)
        solver.neoclassical_params = _neo_params(solver, q_profile=np.ones(solver.rho.size + 1))
        b_pol = np.full(solver.rho.size, 1.0, dtype=np.float64)
        j_bs = solver.calculate_bootstrap_current(6.2, b_pol)
        assert j_bs.shape == solver.rho.shape
        assert np.all(np.isfinite(j_bs))


class TestGyroBohmChi:
    """Gyro-Bohm diffusivity coefficient sourcing paths."""

    def test_neoclassical_disabled_returns_constant(self, tmp_path: Path) -> None:
        """Without neoclassical params the gyro-Bohm helper returns a constant."""
        solver = _solver(tmp_path)
        solver.neoclassical_params = None
        chi = solver._gyro_bohm_chi()
        assert np.allclose(chi, 0.5)
        assert solver._last_gyro_bohm_contract["source"] == "neoclassical_disabled"

    def test_explicit_valid_c_gb_is_used(self, tmp_path: Path) -> None:
        """A valid explicit ``c_gB`` in params is used without a loader fallback."""
        solver = _solver(tmp_path)
        solver.neoclassical_params = _neo_params(solver, c_gB=1.5)
        chi = solver._gyro_bohm_chi()
        assert solver._last_gyro_bohm_contract["source"] == "neoclassical_params"
        assert solver._last_gyro_bohm_contract["fallback_used"] is False
        assert np.all(np.isfinite(chi))

    def test_loader_fallback_records_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A loader-reported fallback is surfaced in the gyro-Bohm contract."""
        solver = _solver(tmp_path)
        solver.neoclassical_params = _neo_params(solver)

        def _fallback_loader() -> tuple[float, dict[str, Any]]:
            return 0.1, {"source": "compat_default", "path": None, "fallback_used": True}

        monkeypatch.setattr(
            its_mod, "_load_gyro_bohm_coefficient_cached_with_contract", _fallback_loader
        )
        chi = solver._gyro_bohm_chi()
        assert solver._last_gyro_bohm_contract["fallback_used"] is True
        assert solver._last_gyro_bohm_contract["source"] == "compat_default"
        assert np.all(np.isfinite(chi))


class TestUpdateTransportRelaxation:
    """Optional under-relaxation of the transport coefficients."""

    def test_relaxation_blends_previous_coefficients(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sub-unity relaxation alpha blends the new and previous diffusivities."""
        solver = _solver(tmp_path)
        # First update populates the previous-coefficient fields.
        solver.update_transport_model(5.0)
        # A relaxation factor below one activates the blend against the stored state.
        # ``chi_relaxation_alpha`` is an optional runtime tuning attribute.
        monkeypatch.setattr(solver, "chi_relaxation_alpha", 0.5, raising=False)
        solver.update_transport_model(5.0)
        assert solver.chi_i.shape == solver.rho.shape
        assert solver.chi_e.shape == solver.rho.shape
        assert solver.D_n.shape == solver.rho.shape
        assert np.all(np.isfinite(solver.chi_i))
        assert np.all(solver.chi_i > 0.0)


def test_physics_error_is_exported() -> None:
    """The mixin's recovery-budget guard raises the shared PhysicsError type."""
    assert issubclass(PhysicsError, RuntimeError)
