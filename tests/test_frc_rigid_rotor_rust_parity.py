#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rust Parity Tests
"""Parity tests for the Rust FRC analytical solver exposed through PyO3."""

from __future__ import annotations

from collections.abc import Callable
import sys
from pathlib import Path
from typing import Any, TypeAlias, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scpn_fusion.core.frc_rigid_rotor import (
    ELEMENTARY_CHARGE_C,
    MU_0,
    RigidRotorFRCInputs,
    solve_frc_equilibrium,
)

FloatArray: TypeAlias = NDArray[np.float64]
_F = TypeVar("_F", bound=Callable[..., object])
_TYPED_GIVEN = cast(Callable[..., Callable[[_F], _F]], given)
_TYPED_SETTINGS = cast(Callable[..., Callable[[_F], _F]], settings)

try:
    scpn_fusion_rs = cast(Any, __import__("scpn_fusion_rs"))
    HAS_RUST = hasattr(scpn_fusion_rs, "py_solve_frc_equilibrium")
    HAS_ROTATING_STATUS = hasattr(scpn_fusion_rs, "py_rotating_frc_bvp_acceptance_status")
except ImportError:
    scpn_fusion_rs = cast(Any, None)
    HAS_RUST = False
    HAS_ROTATING_STATUS = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust FRC extension not available")


def _parity_cases() -> list[tuple[float | None, int, float, float]]:
    """Return deterministic MIF/FRC no-rotation parity cases spanning scale."""
    return [
        (0.012, 33, 2.75, 0.16),
        (0.014, 65, 3.25, 0.18),
        (0.016, 129, 3.75, 0.19),
        (0.018, 257, 4.25, 0.20),
        (0.020, 33, 4.75, 0.21),
        (0.022, 65, 5.25, 0.22),
        (0.024, 129, 5.75, 0.23),
        (0.026, 257, 6.25, 0.24),
        (0.028, 33, 6.75, 0.25),
        (0.030, 65, 7.25, 0.26),
        (0.032, 129, 7.75, 0.27),
        (0.034, 257, 8.25, 0.28),
        (None, 33, 3.50, 0.17),
        (None, 65, 5.00, 0.20),
        (None, 129, 6.50, 0.24),
        (None, 257, 8.00, 0.28),
    ]


def _case(
    delta: float | None, grid_points: int, b_ext: float, r_s: float
) -> tuple[RigidRotorFRCInputs, FloatArray]:
    t_i_ev = 10_000.0
    t_e_ev = 5_000.0
    n0 = b_ext**2 / (2.0 * MU_0) / ((t_i_ev + t_e_ev) * ELEMENTARY_CHARGE_C)
    inputs = RigidRotorFRCInputs(
        n0=n0,
        T_i_eV=t_i_ev,
        T_e_eV=t_e_ev,
        theta_dot=0.0,
        R_s=r_s,
        B_ext=b_ext,
        delta=delta,
    )
    return inputs, np.linspace(0.0, 2.0 * r_s, grid_points)


_frc_delta = st.floats(
    min_value=0.010,
    max_value=0.045,
    allow_nan=False,
    allow_infinity=False,
)
_frc_b_ext = st.floats(
    min_value=2.5,
    max_value=9.0,
    allow_nan=False,
    allow_infinity=False,
)
_frc_r_s = st.floats(
    min_value=0.14,
    max_value=0.32,
    allow_nan=False,
    allow_infinity=False,
)
_frc_grid = st.integers(min_value=65, max_value=257)


def test_rust_frc_matches_python_reference() -> None:
    for delta, grid_points, b_ext, r_s in _parity_cases():
        inputs, rho = _case(delta, grid_points, b_ext, r_s)
        python_state = solve_frc_equilibrium(inputs, rho)
        rust_state = scpn_fusion_rs.py_solve_frc_equilibrium(
            rho,
            inputs.n0,
            inputs.T_i_eV,
            inputs.T_e_eV,
            inputs.theta_dot,
            inputs.R_s,
            inputs.B_ext,
            inputs.delta,
            1.0e-10,
        )

        np.testing.assert_allclose(rust_state["rho"], python_state.rho, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(rust_state["B_z"], python_state.B_z, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(rust_state["B_theta"], python_state.B_theta, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            rust_state["J_theta"], python_state.J_theta, rtol=1.0e-12, atol=1.0e-6
        )
        np.testing.assert_allclose(rust_state["psi"], python_state.psi, rtol=1.0e-13, atol=1.0e-13)
        np.testing.assert_allclose(
            rust_state["psi_normalized"],
            python_state.psi_normalized,
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(rust_state["p"], python_state.p, rtol=1.0e-12, atol=1.0e-6)
        np.testing.assert_allclose(
            rust_state["density_m3"], python_state.density_m3, rtol=1.0e-10, atol=1.0e7
        )
        np.testing.assert_allclose(
            rust_state["beta"], python_state.beta, rtol=1.0e-12, atol=1.0e-12
        )
        np.testing.assert_allclose(
            rust_state["pressure_balance_residual"],
            python_state.pressure_balance_residual,
            rtol=0.0,
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            rust_state["pressure_gradient_analytic_Pa_m"],
            python_state.pressure_gradient_analytic_Pa_m,
            rtol=1.0e-12,
            atol=1.0e-4,
        )
        np.testing.assert_allclose(
            rust_state["pressure_gradient_residual"],
            python_state.pressure_gradient_residual,
            rtol=1.0e-10,
            atol=1.0e-2,
        )
        np.testing.assert_allclose(
            rust_state["flux_derivative_residual"],
            python_state.flux_derivative_residual,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            rust_state["ampere_residual"],
            python_state.ampere_residual,
            rtol=0.0,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            rust_state["force_balance_residual"],
            python_state.force_balance_residual,
            rtol=1.0e-10,
            atol=1.0e-2,
        )

        assert rust_state["model"] == python_state.model
        assert rust_state["converged"] is python_state.converged
        assert int(rust_state["separatrix_index"]) == python_state.separatrix_index
        assert bool(rust_state["field_reversal_passed"]) is python_state.field_reversal_passed
        assert float(rust_state["R_null"]) == pytest.approx(python_state.R_null, abs=1.0e-12)
        assert float(rust_state["target_separatrix_radius_m"]) == pytest.approx(
            python_state.target_separatrix_radius_m,
            abs=1.0e-12,
        )
        assert float(rust_state["separatrix_radius_error_m"]) == pytest.approx(
            python_state.separatrix_radius_error_m,
            abs=1.0e-12,
        )
        assert float(rust_state["s_parameter"]) == pytest.approx(
            python_state.s_parameter, rel=1.0e-12
        )
        assert float(rust_state["delta"]) == pytest.approx(python_state.delta, rel=1.0e-13)
        assert float(rust_state["energy_J"]) == pytest.approx(python_state.energy_J, rel=1.0e-12)
        assert float(rust_state["psi_axis_Wb"]) == pytest.approx(
            python_state.psi_axis_Wb, abs=1.0e-14
        )
        assert float(rust_state["psi_separatrix_Wb"]) == pytest.approx(
            python_state.psi_separatrix_Wb,
            rel=1.0e-12,
        )
        assert float(rust_state["psi_normalized_axis_error"]) == pytest.approx(
            python_state.psi_normalized_axis_error,
            abs=1.0e-14,
        )
        assert float(rust_state["psi_normalized_separatrix"]) == pytest.approx(
            python_state.psi_normalized_separatrix,
            abs=1.0e-12,
        )
        assert float(rust_state["psi_normalized_separatrix_error"]) == pytest.approx(
            python_state.psi_normalized_separatrix_error,
            abs=1.0e-12,
        )
        assert float(rust_state["psi_normalized_residual_linf"]) == pytest.approx(
            python_state.psi_normalized_residual_linf,
            abs=1.0e-12,
        )
        assert (
            bool(rust_state["psi_normalized_monotonic_passed"])
            is python_state.psi_normalized_monotonic_passed
        )
        assert (
            bool(rust_state["psi_normalized_bounds_passed"])
            is python_state.psi_normalized_bounds_passed
        )
        assert float(rust_state["pressure_balance_ratio"]) == pytest.approx(
            python_state.pressure_balance_ratio,
            rel=1.0e-12,
        )
        assert float(rust_state["pressure_balance_residual_linf"]) == pytest.approx(
            python_state.pressure_balance_residual_linf,
            abs=1.0e-12,
        )
        assert float(rust_state["pressure_balance_residual_l2"]) == pytest.approx(
            python_state.pressure_balance_residual_l2,
            abs=1.0e-12,
        )
        assert float(rust_state["pressure_gradient_residual_linf"]) == pytest.approx(
            python_state.pressure_gradient_residual_linf,
            rel=1.0e-10,
        )
        assert float(rust_state["pressure_gradient_residual_l2"]) == pytest.approx(
            python_state.pressure_gradient_residual_l2,
            rel=1.0e-10,
        )
        assert float(rust_state["peak_pressure_pa"]) == pytest.approx(
            python_state.peak_pressure_pa,
            rel=1.0e-12,
        )
        assert float(rust_state["density_peak_m3"]) == pytest.approx(
            python_state.density_peak_m3,
            rel=1.0e-12,
        )
        assert float(rust_state["input_density_m3"]) == pytest.approx(
            python_state.input_density_m3,
            rel=1.0e-12,
        )
        assert float(rust_state["central_density_residual_m3"]) == pytest.approx(
            python_state.central_density_residual_m3,
            abs=max(python_state.input_density_m3, 1.0) * 1.0e-12,
        )
        assert float(rust_state["central_density_relative_error"]) == pytest.approx(
            python_state.central_density_relative_error,
            abs=1.0e-12,
        )
        assert float(rust_state["beta_peak"]) == pytest.approx(
            python_state.beta_peak,
            abs=1.0e-12,
        )
        assert float(rust_state["beta_separatrix_average"]) == pytest.approx(
            python_state.beta_separatrix_average,
            rel=1.0e-12,
        )
        assert float(rust_state["particle_line_density_m1"]) == pytest.approx(
            python_state.particle_line_density_m1,
            rel=1.0e-12,
        )
        assert float(rust_state["separatrix_pressure_energy_J_m"]) == pytest.approx(
            python_state.separatrix_pressure_energy_J_m,
            rel=1.0e-12,
        )
        assert float(rust_state["separatrix_magnetic_deficit_energy_J_m"]) == pytest.approx(
            python_state.separatrix_magnetic_deficit_energy_J_m,
            rel=1.0e-12,
        )
        assert float(rust_state["separatrix_energy_closure_relative_error"]) == pytest.approx(
            python_state.separatrix_energy_closure_relative_error,
            abs=1.0e-12,
        )
        assert float(rust_state["input_thermal_pressure_pa"]) == pytest.approx(
            python_state.input_thermal_pressure_pa,
            rel=1.0e-12,
        )
        assert float(rust_state["thermal_pressure_ratio"]) == pytest.approx(
            python_state.thermal_pressure_ratio,
            rel=1.0e-12,
        )
        assert float(rust_state["flux_derivative_residual_linf"]) == pytest.approx(
            python_state.flux_derivative_residual_linf,
            abs=1.0e-12,
        )
        assert float(rust_state["flux_derivative_residual_l2"]) == pytest.approx(
            python_state.flux_derivative_residual_l2,
            abs=1.0e-12,
        )
        assert float(rust_state["peak_j_theta_A_m2"]) == pytest.approx(
            python_state.peak_j_theta_A_m2,
            rel=1.0e-12,
        )
        assert float(rust_state["ampere_residual_linf"]) == pytest.approx(
            python_state.ampere_residual_linf,
            abs=1.0e-12,
        )
        assert float(rust_state["ampere_residual_l2"]) == pytest.approx(
            python_state.ampere_residual_l2,
            abs=1.0e-12,
        )
        assert float(rust_state["separatrix_bz_gradient_T_m"]) == pytest.approx(
            python_state.separatrix_bz_gradient_T_m,
            rel=1.0e-12,
        )
        assert float(rust_state["separatrix_expected_bz_gradient_T_m"]) == pytest.approx(
            python_state.separatrix_expected_bz_gradient_T_m,
            rel=1.0e-12,
        )
        assert float(rust_state["separatrix_gradient_relative_error"]) == pytest.approx(
            python_state.separatrix_gradient_relative_error,
            abs=1.0e-12,
        )
        assert float(rust_state["separatrix_current_density_A_m2"]) == pytest.approx(
            python_state.separatrix_current_density_A_m2,
            rel=1.0e-12,
        )
        assert float(rust_state["separatrix_expected_current_density_A_m2"]) == pytest.approx(
            python_state.separatrix_expected_current_density_A_m2,
            rel=1.0e-12,
        )
        assert float(rust_state["separatrix_current_density_relative_error"]) == pytest.approx(
            python_state.separatrix_current_density_relative_error,
            abs=1.0e-12,
        )
        assert float(rust_state["sheet_current_integral_A_m"]) == pytest.approx(
            python_state.sheet_current_integral_A_m,
            rel=1.0e-12,
        )
        assert float(rust_state["expected_sheet_current_integral_A_m"]) == pytest.approx(
            python_state.expected_sheet_current_integral_A_m,
            rel=1.0e-12,
        )
        assert float(rust_state["sheet_current_integral_relative_error"]) == pytest.approx(
            python_state.sheet_current_integral_relative_error,
            abs=1.0e-12,
        )
        assert float(rust_state["force_balance_residual_linf"]) == pytest.approx(
            python_state.force_balance_residual_linf,
            rel=1.0e-10,
        )
        assert float(rust_state["force_balance_residual_l2"]) == pytest.approx(
            python_state.force_balance_residual_l2,
            rel=1.0e-10,
        )


@_TYPED_GIVEN(delta=_frc_delta, b_ext=_frc_b_ext, r_s=_frc_r_s, grid_points=_frc_grid)
@_TYPED_SETTINGS(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_rust_energy_invariants_match_python_for_generated_no_rotation_decks(
    delta: float,
    b_ext: float,
    r_s: float,
    grid_points: int,
) -> None:
    inputs, rho = _case(delta, grid_points, b_ext, r_s)
    python_state = solve_frc_equilibrium(inputs, rho)

    rust_state = scpn_fusion_rs.py_solve_frc_equilibrium(
        rho,
        inputs.n0,
        inputs.T_i_eV,
        inputs.T_e_eV,
        inputs.theta_dot,
        inputs.R_s,
        inputs.B_ext,
        inputs.delta,
        1.0e-10,
    )

    assert float(rust_state["energy_J"]) == pytest.approx(
        python_state.energy_J,
        rel=1.0e-12,
    )
    assert float(rust_state["pressure_balance_ratio"]) == pytest.approx(
        python_state.pressure_balance_ratio,
        rel=1.0e-12,
    )
    assert float(rust_state["separatrix_pressure_energy_J_m"]) == pytest.approx(
        python_state.separatrix_pressure_energy_J_m,
        rel=1.0e-12,
    )
    assert float(rust_state["separatrix_magnetic_deficit_energy_J_m"]) == pytest.approx(
        python_state.separatrix_magnetic_deficit_energy_J_m,
        rel=1.0e-12,
    )
    assert float(rust_state["separatrix_energy_closure_relative_error"]) == pytest.approx(
        python_state.separatrix_energy_closure_relative_error,
        abs=1.0e-12,
    )
    np.testing.assert_allclose(
        rust_state["psi_normalized"],
        python_state.psi_normalized,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_rust_frc_rejects_rotating_bvp_until_implemented() -> None:
    inputs, rho = _case(0.02, 65, 5.0, 0.20)
    with pytest.raises(ValueError, match="rotating rigid-rotor BVP"):
        scpn_fusion_rs.py_solve_frc_equilibrium(
            rho,
            inputs.n0,
            inputs.T_i_eV,
            inputs.T_e_eV,
            1.0,
            inputs.R_s,
            inputs.B_ext,
            inputs.delta,
            1.0e-10,
        )


def test_rust_rotating_bvp_status_matches_python_fail_closed_boundary() -> None:
    if not HAS_ROTATING_STATUS:
        pytest.skip("local Rust extension has not been rebuilt with rotating BVP status binding")

    status = scpn_fusion_rs.py_rotating_frc_bvp_acceptance_status()

    assert status["status"] == "blocked_missing_verified_steinhauer_rotating_closure"
    assert status["accepted_contract"] == "steinhauer_2011_no_rotation_analytical"
    assert status["rotating_bvp_implemented"] is False
    assert status["solver_action"] == "raise_not_implemented_for_nonzero_theta_dot"
    assert status["required_reference"] == "Steinhauer 2011 Section II.B plus Figure 3 closure"
    assert "Romero 2018" in status["non_closing_references"]
    assert "Slough 2011 Fig. 5" in status["non_closing_references"]
    assert "not a rotating-BVP solver certification" in status["claim_boundary"]
