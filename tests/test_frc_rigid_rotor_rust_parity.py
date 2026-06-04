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

import sys
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scpn_fusion.core.frc_rigid_rotor import ELEMENTARY_CHARGE_C, MU_0, RigidRotorFRCInputs, solve_frc_equilibrium

FloatArray: TypeAlias = NDArray[np.float64]

try:
    scpn_fusion_rs = cast(Any, __import__("scpn_fusion_rs"))
    HAS_RUST = hasattr(scpn_fusion_rs, "py_solve_frc_equilibrium")
except ImportError:
    scpn_fusion_rs = cast(Any, None)
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust FRC extension not available")


def _case(delta: float | None, grid_points: int, b_ext: float, r_s: float) -> tuple[RigidRotorFRCInputs, FloatArray]:
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


def test_rust_frc_matches_python_reference() -> None:
    cases: list[tuple[float | None, int, float, float]] = [
        (0.015, 33, 3.0, 0.18),
        (0.020, 65, 5.0, 0.20),
        (0.025, 129, 7.5, 0.24),
        (None, 257, 5.0, 0.20),
    ]
    for delta, grid_points, b_ext, r_s in cases:
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
        np.testing.assert_allclose(rust_state["J_theta"], python_state.J_theta, rtol=1.0e-12, atol=1.0e-6)
        np.testing.assert_allclose(rust_state["psi"], python_state.psi, rtol=1.0e-13, atol=1.0e-13)
        np.testing.assert_allclose(rust_state["p"], python_state.p, rtol=1.0e-12, atol=1.0e-6)
        np.testing.assert_allclose(rust_state["density_m3"], python_state.density_m3, rtol=1.0e-10, atol=1.0e7)
        np.testing.assert_allclose(rust_state["beta"], python_state.beta, rtol=1.0e-12, atol=1.0e-12)
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
            atol=1.0e-12,
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
        assert float(rust_state["s_parameter"]) == pytest.approx(python_state.s_parameter, rel=1.0e-12)
        assert float(rust_state["delta"]) == pytest.approx(python_state.delta, rel=1.0e-13)
        assert float(rust_state["energy_J"]) == pytest.approx(python_state.energy_J, rel=1.0e-12)
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
