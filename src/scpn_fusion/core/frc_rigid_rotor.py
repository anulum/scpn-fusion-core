# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rigid-Rotor Equilibrium
"""Field-reversed-configuration rigid-rotor equilibrium helpers.

This package implements two verified rigid-rotor FRC equilibrium closures:

* the Steinhauer no-rotation analytical limit (``theta_dot == 0``), the accepted
  magnetostatic pressure-balance contract used across the fusion pipeline; and
* the Rostoker & Qerushi (2002) one-dimensional, one-ion rotating rigid-rotor
  closure (``theta_dot != 0``), which adds the verified centrifugal source
  ``rho * omega**2 * r`` to the radial force balance and reduces bit-exactly to
  the no-rotation contract as ``theta_dot -> 0``.

The rotating closure is grounded in the source-verified governing equations
recorded in
``docs/internal/reference_papers/frc/rotating_rigid_rotor_verified_closure_2026-07-01.md``
(Rostoker & Qerushi 2002, Phys. Plasmas 9, 3057, reproduced in US 6,664,740 B2;
non-rotating limit cross-checked against arXiv:2010.05493). Verbatim Steinhauer
2011 Figure 3 digitised parity is deliberately not claimed here and remains a
separate external-parity gate.

This module is the public facade of the ``frc_rigid_rotor`` package. The
implementation is split by responsibility across four submodules and re-exported
here so the historical import surface stays byte-identical:

* :mod:`frc_rigid_rotor_contracts` — physical constants and data contracts;
* :mod:`frc_rigid_rotor_closures` — shared analytical closures and accessors;
* :mod:`frc_rigid_rotor_solver` — the equilibrium solver and input guards;
* :mod:`frc_rigid_rotor_validation` — the equilibrium acceptance validation.
"""

from __future__ import annotations

from .frc_rigid_rotor_closures import (
    ampere_residual,
    beta_profile,
    density_profile,
    flux_derivative_residual,
    force_balance_residual,
    null_radius,
    pressure_balance_residual,
    pressure_gradient_residual,
    psi_normalized_profile,
    s_parameter,
)
from .frc_rigid_rotor_contracts import (
    ATOMIC_MASS_KG,
    DEUTERIUM_MASS_AMU,
    ELEMENTARY_CHARGE_C,
    FRCEquilibriumState,
    FRCValidationReport,
    FloatArray,
    MU_0,
    ROTATING_FRC_BVP_CLAIM_BOUNDARY,
    ROTATING_FRC_BVP_NON_CLOSING_REFERENCES,
    ROTATING_FRC_BVP_REQUIRED_REFERENCE,
    ROTATING_FRC_BVP_ROTATING_REFERENCE,
    ROTATING_FRC_BVP_SOLVER_ACTION,
    ROTATING_FRC_BVP_STATUS,
    RigidRotorFRCInputs,
    rotating_frc_bvp_acceptance_status,
)
from .frc_rigid_rotor_solver import (
    frc_no_rotation_jax_observables,
    ion_gyroradius_m,
    solve_frc_equilibrium,
)
from .frc_rigid_rotor_validation import validate_equilibrium

__all__ = [
    "ATOMIC_MASS_KG",
    "DEUTERIUM_MASS_AMU",
    "ELEMENTARY_CHARGE_C",
    "FRCEquilibriumState",
    "FRCValidationReport",
    "FloatArray",
    "MU_0",
    "ROTATING_FRC_BVP_CLAIM_BOUNDARY",
    "ROTATING_FRC_BVP_NON_CLOSING_REFERENCES",
    "ROTATING_FRC_BVP_REQUIRED_REFERENCE",
    "ROTATING_FRC_BVP_ROTATING_REFERENCE",
    "ROTATING_FRC_BVP_SOLVER_ACTION",
    "ROTATING_FRC_BVP_STATUS",
    "RigidRotorFRCInputs",
    "ampere_residual",
    "beta_profile",
    "density_profile",
    "flux_derivative_residual",
    "force_balance_residual",
    "frc_no_rotation_jax_observables",
    "ion_gyroradius_m",
    "null_radius",
    "pressure_balance_residual",
    "pressure_gradient_residual",
    "psi_normalized_profile",
    "rotating_frc_bvp_acceptance_status",
    "s_parameter",
    "solve_frc_equilibrium",
    "validate_equilibrium",
]
