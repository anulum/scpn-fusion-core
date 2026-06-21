# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Integrated Transport Solver Shared State Interface
"""Shared typed state interface for the integrated transport solver mixins.

``TransportSolver`` is composed from the initialization, model and runtime mixins on
top of :class:`FusionKernel`. The mixins reference one another's transport state
(profiles, grids, species, recovery counters) and methods (profile evolution, implicit
diffusion, recovery accounting), as well as kernel-provided geometry. Declaring that
shared surface once here lets every mixin be strict-type-checked in isolation.

Kernel-provided members (``cfg``, ``Psi``, ``RR``, ``dR``, ``dZ``,
``solve_equilibrium``, ``find_x_point``) are declared as bare annotations only — a
class-level annotation creates no runtime attribute, so the real ``FusionKernel``
implementation (which follows this base in the method resolution order) is used at
runtime while the access still type-checks. Transport members defined by the mixins
themselves are declared as attributes or as ``NotImplementedError`` stubs, which the
defining mixin overrides ahead of this base in the resolution order.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class TransportSolverState:
    """Typed declaration of the composed-solver state shared across the mixins."""

    # Kernel-provided members (annotation-only — never shadow FusionKernel at runtime).
    cfg: dict[str, Any]
    Psi: FloatArray
    RR: FloatArray
    dR: float
    dZ: float
    solve_equilibrium: Callable[..., Any]
    find_x_point: Callable[[FloatArray], tuple[tuple[float, float], float]]

    # Radial grid and kinetic profiles (set in the initialization mixin).
    rho: FloatArray
    drho: float
    nr: int
    Te: FloatArray
    Ti: FloatArray
    ne: FloatArray
    n_impurity: FloatArray
    n_D: FloatArray
    n_T: FloatArray
    n_He: FloatArray
    T_edge_keV: float

    # Species, pedestal and auxiliary-heating configuration.
    multi_ion: bool
    D_species: float
    tau_He_factor: float
    pedestal_model: Any
    neoclassical_params: dict[str, Any] | None
    aux_heating_profile_width: float
    aux_heating_electron_fraction: float

    # Numerical-recovery accounting.
    _last_numerical_recovery_count: int
    _last_numerical_recovery_breakdown: dict[str, Any]

    def _record_recovery(self, label: str, count: int) -> None:
        """Record a numerical-recovery event against the per-step budget."""
        raise NotImplementedError

    def _enforce_recovery_budget(
        self,
        *,
        enforce_numerical_recovery: bool,
        max_numerical_recoveries: int | None,
    ) -> None:
        """Raise if the numerical-recovery count exceeds the configured budget."""
        raise NotImplementedError

    def update_transport_model(self, P_aux: float) -> None:
        """Refresh transport coefficients for the current profiles and auxiliary power."""
        raise NotImplementedError

    def evolve_profiles(
        self,
        dt: float,
        P_aux: float,
        enforce_conservation: bool = False,
        *,
        enforce_numerical_recovery: bool = False,
        max_numerical_recoveries: int | None = None,
    ) -> tuple[float, float]:
        """Advance the kinetic profiles by one transport timestep."""
        raise NotImplementedError

    def _evolve_impurity(self, dt: float) -> None:
        """Advance the impurity density profile by one timestep."""
        raise NotImplementedError

    def _rho_volume_element(self) -> FloatArray:
        """Return the per-cell flux-surface volume element on the rho grid."""
        raise NotImplementedError

    def compute_confinement_time(self, P_loss_MW: float) -> float:
        """Compute the energy confinement time for the current stored energy."""
        raise NotImplementedError

    def calculate_bootstrap_current(self, R0: float, B_pol: FloatArray) -> FloatArray:
        """Compute the bootstrap current density profile."""
        raise NotImplementedError

    def _explicit_diffusion_rhs(self, T: FloatArray, chi: FloatArray) -> FloatArray:
        """Evaluate the explicit radial diffusion operator on a profile."""
        raise NotImplementedError

    def _build_cn_tridiag(
        self, chi: FloatArray, dt: float
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Build Crank-Nicolson tridiagonal coefficients for the implicit solve."""
        raise NotImplementedError

    @staticmethod
    def _thomas_solve(a: FloatArray, b: FloatArray, c: FloatArray, d: FloatArray) -> FloatArray:
        """Solve a tridiagonal system via the Thomas algorithm."""
        raise NotImplementedError

    def _get_neural_transport_model(self) -> Any:
        """Return the lazily-constructed neural transport surrogate model."""
        raise NotImplementedError

    def _select_neural_ood_indices(
        self,
        max_abs_z_profile: FloatArray,
        *,
        sigma_threshold: float,
        max_points: int,
    ) -> tuple[BoolArray, list[int]]:
        """Select out-of-distribution radial indices for the neural surrogate."""
        raise NotImplementedError

    def _summarize_coarse_transport_channels(
        self,
        chi_e_profile: FloatArray,
        chi_i_profile: FloatArray,
    ) -> tuple[str, dict[str, int], dict[str, float]]:
        """Summarise coarse transport-channel diagnostics for logging."""
        raise NotImplementedError


__all__ = ["TransportSolverState"]
