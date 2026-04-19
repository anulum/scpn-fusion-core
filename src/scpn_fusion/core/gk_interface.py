# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Gyrokinetic Solver Interface
"""
Abstract base class for external and native gyrokinetic solvers.

Defines the universal input/output dataclasses (GKLocalParams, GKOutput)
and the solver contract (GKSolverBase) that all concrete implementations
must satisfy.  The design follows the TGLF-10 parameterisation extended
with Miller geometry fields for flux-tube solvers.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class GKLocalParams:
    """Local plasma parameters at a single flux surface.

    The first 11 fields match the GyrokineticsParams convention already
    used by `gyrokinetic_transport.py`.  The remaining fields add Miller
    geometry and dimensional quantities needed by external GK codes.
    """

    # Normalised gradients
    R_L_Ti: float
    R_L_Te: float
    R_L_ne: float

    # Magnetic geometry
    q: float
    s_hat: float
    alpha_MHD: float = 0.0

    # Ratios / dimensionless
    Te_Ti: float = 1.0
    Z_eff: float = 1.5
    nu_star: float = 0.1
    beta_e: float = 0.01
    epsilon: float = 0.1  # r / R

    # Miller shaping
    kappa: float = 1.0
    delta: float = 0.0

    # Dimensional
    rho: float = 0.5  # normalised toroidal flux
    R0: float = 6.2  # major radius [m]
    a: float = 2.0  # minor radius [m]
    B0: float = 5.3  # toroidal field [T]
    n_e: float = 10.0  # electron density [10^19 m^-3]
    T_e_keV: float = 8.0  # electron temperature [keV]
    T_i_keV: float = 8.0  # ion temperature [keV]


@dataclass
class GKOutput:
    """Gyrokinetic solver output for a single flux surface.

    Fluxes are in physical units [m^2/s].  Growth rate arrays are
    normalised to c_s / a.
    """

    chi_i: float  # ion thermal diffusivity [m^2/s]
    chi_e: float  # electron thermal diffusivity [m^2/s]
    D_e: float  # particle diffusivity [m^2/s]
    D_i: float = 0.0  # ion particle diffusivity [m^2/s]

    gamma: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    omega_r: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    k_y: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))

    dominant_mode: str = "stable"  # ITG / TEM / ETG / stable
    converged: bool = True


class GKSolverBase(abc.ABC):
    """Abstract base for gyrokinetic solvers.

    Concrete subclasses must implement ``prepare_input``, ``run``,
    and ``is_available``.
    """

    @abc.abstractmethod
    def prepare_input(self, params: GKLocalParams) -> Path:
        """Write solver-specific input deck to a temporary directory.

        Returns the path to the input file or directory.
        """

    @abc.abstractmethod
    def run(self, input_path: Path, *, timeout_s: float = 30.0) -> GKOutput:
        """Execute the solver and parse output into GKOutput."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if the solver binary/library is installed."""

    def run_from_params(self, params: GKLocalParams, *, timeout_s: float = 30.0) -> GKOutput:
        """Convenience: prepare + run in one call."""
        input_path = self.prepare_input(params)
        return self.run(input_path, timeout_s=timeout_s)
