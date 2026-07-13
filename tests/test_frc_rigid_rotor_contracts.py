# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rigid-Rotor Equilibrium Contracts
"""Unit contracts for the FRC rigid-rotor data layer: constants, the empty-array
default factory, the input dataclass, and the rotating-closure acceptance status.
"""

from __future__ import annotations

import numpy as np

from scpn_fusion.core.frc_rigid_rotor_contracts import (
    ATOMIC_MASS_KG,
    DEUTERIUM_MASS_AMU,
    ELEMENTARY_CHARGE_C,
    MU_0,
    ROTATING_FRC_BVP_STATUS,
    RigidRotorFRCInputs,
    _empty_float_array,
    rotating_frc_bvp_acceptance_status,
)


def test_physical_constants_have_reference_values() -> None:
    assert 4.0 * np.pi * 1e-7 == MU_0
    assert ELEMENTARY_CHARGE_C == 1.602176634e-19
    assert ATOMIC_MASS_KG == 1.66053906660e-27
    assert DEUTERIUM_MASS_AMU == 2.014


def test_empty_float_array_is_empty_float64() -> None:
    arr = _empty_float_array()
    assert arr.dtype == np.float64
    assert arr.shape == (0,)


def test_inputs_dataclass_defaults_delta_to_none() -> None:
    inputs = RigidRotorFRCInputs(
        n0=1.0e19,
        T_i_eV=200.0,
        T_e_eV=180.0,
        theta_dot=0.0,
        R_s=0.3,
        B_ext=0.5,
    )
    assert inputs.delta is None
    assert inputs.n0 == 1.0e19


def test_acceptance_status_reports_verified_rotating_closure() -> None:
    status = rotating_frc_bvp_acceptance_status()
    assert status["status"] == ROTATING_FRC_BVP_STATUS
    assert status["rotating_bvp_implemented"] is True
    assert status["reduces_to_no_rotation_contract"] is True
    # The Steinhauer figure-3 parity is deliberately NOT claimed here.
    assert status["steinhauer_figure3_parity_claimed"] is False
    assert "claim_boundary" in status
