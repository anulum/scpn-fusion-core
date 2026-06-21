# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Nonlinear GK Solver Shared State Interface
"""Shared typed state interface for the nonlinear gyrokinetic solver mixins.

``NonlinearGKSolver`` is composed from :class:`NonlinearGKSetupMixin`,
:class:`NonlinearGKOperatorsMixin` and :class:`NonlinearGKTimeMixin`, which freely
reference one another's attributes (grids, geometry, species) and methods (field
solves, brackets, RHS). Declaring that shared surface once here lets every mixin be
strict-type-checked in isolation; the concrete attributes are assigned by the setup
mixin and the concrete methods are implemented by the operator/time mixins.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core._gk_nonlinear_types import (
    NonlinearGKConfig,
    NonlinearGKInvariantDiagnostics,
    NonlinearGKState,
)
from scpn_fusion.core.gk_geometry import MillerGeometry
from scpn_fusion.core.gk_species import GKSpecies

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]


class NonlinearGKSolverState:
    """Typed declaration of the composed-solver state shared across the mixins."""

    cfg: NonlinearGKConfig

    # Spectral and velocity-space grids (set in NonlinearGKSetupMixin._setup_grids).
    kx: FloatArray
    ky: FloatArray
    kx_grid: FloatArray
    ky_grid: FloatArray
    kperp2: FloatArray
    theta: FloatArray
    dtheta: float
    vpar: FloatArray
    dvpar: float
    mu: FloatArray
    dmu: float
    dealias_mask: BoolArray
    _ball_phase_fwd: ComplexArray
    _ball_phase_bwd: ComplexArray

    # Geometry (set in _setup_geometry).
    geom: MillerGeometry
    b_dot_grad: FloatArray
    kappa_n: FloatArray
    kappa_g: FloatArray
    B_ratio: FloatArray

    # Species and derived scalars (set in _setup_species).
    ion: GKSpecies
    elec: GKSpecies
    c_s: float
    rho_s: float
    rho_ratio: float
    rho_ratio_e: float
    vth_ratio_e: float
    _rh_rate: float
    _ky_zero_5d: BoolArray

    def field_solve(self, f: ComplexArray) -> ComplexArray:
        """Solve the gyrokinetic quasineutrality field equation for phi."""
        raise NotImplementedError

    def ampere_solve(self, f: ComplexArray) -> ComplexArray:
        """Solve the parallel Ampere law for A_parallel."""
        raise NotImplementedError

    def magnetic_compression_solve(self, f: ComplexArray) -> ComplexArray:
        """Solve the perpendicular Ampere law for B_parallel."""
        raise NotImplementedError

    def exb_bracket(self, phi: ComplexArray, f_s: ComplexArray) -> ComplexArray:
        """Compute the dealiased E x B Poisson bracket {phi, f}."""
        raise NotImplementedError

    def rhs(self, state: NonlinearGKState) -> ComplexArray:
        """Evaluate the full nonlinear gyrokinetic right-hand side."""
        raise NotImplementedError

    def validate_state(self, state: NonlinearGKState) -> None:
        """Validate distribution/field shapes, finiteness and contracts."""
        raise NotImplementedError

    def _roll_ballooning(self, f_s: ComplexArray, shift: int) -> ComplexArray:
        """Roll along theta with ballooning kx phase shifts at the boundary."""
        raise NotImplementedError

    def nonlinear_invariant_diagnostics(
        self, state: NonlinearGKState
    ) -> NonlinearGKInvariantDiagnostics:
        """Compute discrete nonlinear free-energy and dealiasing invariants."""
        raise NotImplementedError


__all__ = ["NonlinearGKSolverState"]
