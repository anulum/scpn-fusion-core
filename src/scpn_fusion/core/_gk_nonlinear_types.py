# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Contracts and constants for the nonlinear gyrokinetic solver."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


_E_CHARGE = 1.602176634e-19
_M_PROTON = 1.67262192369e-27


@dataclass
class NonlinearGKConfig:
    """Grid and physics parameters for the nonlinear solver."""

    n_kx: int = 16
    n_ky: int = 16
    n_theta: int = 64
    n_vpar: int = 16
    n_mu: int = 8
    n_species: int = 2

    dt: float = 0.05
    n_steps: int = 5000
    save_interval: int = 100

    Lx: float = 80.0
    Ly: float = 62.83

    vpar_max: float = 3.0
    mu_max: float = 9.0

    dealiasing: str = "2/3"
    hyper_order: int = 4
    hyper_coeff: float = 0.1
    cfl_factor: float = 0.5
    cfl_adapt: bool = True

    collisions: bool = True
    nu_collision: float = 0.01
    collision_model: str = "krook"
    nonlinear: bool = True
    kinetic_electrons: bool = False
    mass_ratio_me_mi: float = 1.0 / 400.0
    implicit_electrons: bool = False
    electromagnetic: bool = False
    beta_e: float = 0.01

    R0: float = 2.78
    a: float = 1.0
    B0: float = 2.0
    q: float = 1.4
    s_hat: float = 0.78

    R_L_Ti: float = 6.9
    R_L_Te: float = 6.9
    R_L_ne: float = 2.2


@dataclass
class NonlinearGKState:
    """Full 5D+1 state of the nonlinear solver."""

    f: NDArray[np.complex128]
    phi: NDArray[np.complex128]
    time: float
    A_par: NDArray[np.complex128] | None = None


@dataclass(frozen=True)
class NonlinearGKPhaseSpaceContract:
    """Explicit five-dimensional nonlinear GK state/grid contract."""

    distribution_shape: tuple[int, int, int, int, int, int]
    field_shape: tuple[int, int, int]
    distribution_axes: tuple[str, str, str, str, str, str]
    field_axes: tuple[str, str, str]
    axis_units: dict[str, str]
    boundary_semantics: dict[str, str]
    dealiasing: str


@dataclass
class NonlinearGKResult:
    """Time-averaged transport and diagnostics."""

    chi_i: float
    chi_e: float
    chi_i_gB: float = 0.0
    Q_i_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    Q_e_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    phi_rms_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    zonal_rms_t: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    time: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    converged: bool = False
    final_state: NonlinearGKState | None = None


@dataclass(frozen=True)
class NonlinearGKInvariantDiagnostics:
    """Discrete nonlinear-operator invariant diagnostics."""

    exb_free_energy_production: float
    exb_relative_free_energy_production: float
    dealiased_high_k_max_abs: float
    finite: bool
    passes: bool


__all__ = [
    "NonlinearGKConfig",
    "NonlinearGKInvariantDiagnostics",
    "NonlinearGKPhaseSpaceContract",
    "NonlinearGKResult",
    "NonlinearGKState",
    "_E_CHARGE",
    "_M_PROTON",
]
