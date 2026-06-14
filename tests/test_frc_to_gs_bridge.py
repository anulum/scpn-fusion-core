# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC to Grad-Shafranov Bridge Tests
"""Regression tests for seeding the 2D GS solver from a 1D FRC equilibrium."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.frc_rigid_rotor import RigidRotorFRCInputs, solve_frc_equilibrium
from scpn_fusion.core.fusion_kernel import FusionKernel


def _config_path(tmp_path: Path) -> Path:
    path = tmp_path / "frc_gs_bridge_config.json"
    path.write_text(
        json.dumps(
            {
                "reactor_name": "FRC-GS-Bridge-Test",
                "grid_resolution": [65, 65],
                "dimensions": {
                    "R_min": 0.7,
                    "R_max": 1.7,
                    "Z_min": -0.8,
                    "Z_max": 0.8,
                },
                "coils": [],
                "physics": {
                    "plasma_current_target": 1.0,
                    "vacuum_permeability": 4.0 * np.pi * 1e-7,
                },
                "solver": {
                    "max_iterations": 1,
                    "convergence_threshold": 1e-4,
                    "relaxation_factor": 0.1,
                    "solver_method": "multigrid",
                    "fail_on_diverge": True,
                },
                "target": {"kappa": 2.0},
            }
        ),
        encoding="utf-8",
    )
    return path


def _frc_state():
    return solve_frc_equilibrium(
        RigidRotorFRCInputs(
            n0=3.0e21,
            T_i_eV=350.0,
            T_e_eV=300.0,
            theta_dot=0.0,
            R_s=0.32,
            B_ext=0.65,
            delta=0.035,
        ),
        np.linspace(0.0, 0.45, 257),
    )


@pytest.mark.parametrize("kappa", [1.0, 2.0, 2.5])
def test_initialize_from_frc_maps_finite_profiles_and_current_target(
    tmp_path: Path, kappa: float
) -> None:
    kernel = FusionKernel(_config_path(tmp_path))
    kernel.initialize_from_frc(_frc_state(), kappa=kappa)

    assert kernel.Psi.shape == (kernel.NZ, kernel.NR)
    assert kernel.J_phi.shape == (kernel.NZ, kernel.NR)
    assert np.all(np.isfinite(kernel.Psi))
    assert np.all(np.isfinite(kernel.J_phi))
    assert kernel.cfg["target"]["kappa"] == pytest.approx(kappa)

    current = float(np.sum(kernel.J_phi)) * kernel.dR * kernel.dZ
    assert kernel.cfg["physics"]["plasma_current_target"] == pytest.approx(current, rel=1e-12)
    assert current > 0.0

    mu0 = kernel.cfg["physics"]["vacuum_permeability"]
    source = -mu0 * kernel.RR * kernel.J_phi
    assert np.all(np.isfinite(source))
    assert np.max(np.abs(source)) < 100.0


def test_initialize_from_frc_uses_zero_current_outside_radial_support(tmp_path: Path) -> None:
    kernel = FusionKernel(_config_path(tmp_path))
    kernel.initialize_from_frc(_frc_state(), kappa=2.0)

    assert np.allclose(kernel.J_phi[:, 0], 0.0, atol=1e-12)
    assert np.allclose(kernel.J_phi[:, -1], 0.0, atol=1e-12)
    assert np.max(np.abs(kernel.Psi[:, 0])) <= np.max(np.abs(kernel.Psi)) + 1e-12
    assert np.max(np.abs(kernel.Psi[:, -1])) <= np.max(np.abs(kernel.Psi)) + 1e-12


def test_initialize_from_frc_kappa_controls_axial_extent(tmp_path: Path) -> None:
    narrow = FusionKernel(_config_path(tmp_path))
    wide = FusionKernel(_config_path(tmp_path))
    state = _frc_state()

    narrow.initialize_from_frc(state, kappa=1.0)
    wide.initialize_from_frc(state, kappa=2.5)

    narrow_profile = np.max(np.abs(narrow.J_phi), axis=1)
    wide_profile = np.max(np.abs(wide.J_phi), axis=1)
    threshold = 0.05 * float(np.max(narrow_profile))
    narrow_extent = int(np.count_nonzero(narrow_profile > threshold))
    wide_extent = int(np.count_nonzero(wide_profile > threshold))

    assert wide_extent > narrow_extent


def test_initialize_from_frc_rejects_invalid_kappa(tmp_path: Path) -> None:
    kernel = FusionKernel(_config_path(tmp_path))

    with pytest.raises(ValueError, match="kappa"):
        kernel.initialize_from_frc(_frc_state(), kappa=0.0)


def test_initialize_from_frc_seed_survives_one_picard_iteration(tmp_path: Path) -> None:
    kernel = FusionKernel(_config_path(tmp_path))
    kernel.initialize_from_frc(_frc_state(), kappa=2.0)

    result = kernel.solve_equilibrium(preserve_initial_state=True)

    assert result["iterations"] <= 1
    assert np.all(np.isfinite(result["psi"]))
    assert np.all(np.isfinite(kernel.J_phi))
    assert np.isfinite(result["residual"])
