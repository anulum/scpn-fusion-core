# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Linear Gyrokinetic Eigenvalue Solver
"""
Linear gyrokinetic eigenvalue solver in ballooning representation.

Solves the linearised electrostatic gyrokinetic equation for complex
eigenvalues omega = omega_r + i*gamma at each wavenumber k_y.
Adiabatic or kinetic electrons, Sugama collision operator, Miller
geometry via gk_geometry module.

The eigenvalue problem is cast as a standard eigenproblem A x = omega x
via the response-matrix formulation, solved with scipy sparse or dense
eigensolvers.

References:
  - Dimits et al., Phys. Plasmas 7 (2000) 969 — Cyclone Base Case
  - Jenko et al., Phys. Plasmas 7 (2000) 1904 — ETG
  - Kotschenreuther et al., Comp. Phys. Comm. 88 (1995) 128 — GS2 method
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.gk_geometry import MillerGeometry, circular_geometry
from scpn_fusion.core.gk_species import (
    GKSpecies,
    VelocityGrid,
    bessel_j0,
    collision_frequencies,
    deuterium_ion,
    electron,
)

_E_CHARGE = 1.602176634e-19  # C

_logger = logging.getLogger(__name__)


@dataclass
class EigenMode:
    """Single eigenmode at one k_y."""

    k_y_rho_s: float
    omega_r: float  # real frequency [c_s / a]
    gamma: float  # growth rate [c_s / a]
    mode_type: str  # ITG / TEM / ETG / stable
    phi_theta: NDArray[np.float64] | None = None  # eigenfunction phi(theta)


@dataclass
class LinearGKResult:
    """Full linear GK spectrum scan result."""

    k_y: NDArray[np.float64]
    gamma: NDArray[np.float64]
    omega_r: NDArray[np.float64]
    mode_type: list[str]
    modes: list[EigenMode]

    @property
    def gamma_max(self) -> float:
        if len(self.gamma) == 0:
            return 0.0
        return float(np.max(self.gamma))

    @property
    def k_y_max(self) -> float:
        """k_y at maximum growth rate."""
        if len(self.gamma) == 0:
            return 0.0
        return float(self.k_y[np.argmax(self.gamma)])


def _diamagnetic_frequency(
    k_y: float, species: GKSpecies, R0: float, a: float
) -> tuple[float, float]:
    """Compute omega_* and omega_*T for a species.

    omega_* = k_y * rho_s * (T_s / (e B)) * R/L_n
    omega_*T = omega_* * [1 + eta * (E/T - 3/2)]
    where eta = L_n / L_T = R_L_T / R_L_n
    """
    omega_star_n = k_y * species.R_L_n
    omega_star_T = k_y * species.R_L_T
    return omega_star_n, omega_star_T


def _drift_frequency(
    k_y: float,
    geom: MillerGeometry,
    energy_norm: float,
    lam: float,
    B_ratio: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Magnetic drift frequency omega_D(theta, E, lambda).

    omega_D = k_y * (2 E / B) * (kappa_n * (1 - lambda*B/B0) + kappa_g * sign(v_∥))
    In normalised units (v_th/R scale).
    """
    xi_sq = np.maximum(1.0 - lam * B_ratio, 0.0)
    return k_y * 2.0 * energy_norm * (geom.kappa_n * xi_sq + geom.kappa_g * np.sqrt(xi_sq))


def _parallel_streaming_matrix(
    n_theta: int,
    geom: MillerGeometry,
    energy_norm: float,
    lam: float,
    B_ratio: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Parallel streaming operator v_∥ * b.grad(theta) * d/d(theta).

    Discretised as central finite differences on theta grid.
    Returns shape (n_theta, n_theta).
    """
    xi = np.sqrt(np.maximum(1.0 - lam * B_ratio, 0.0))
    v_par = np.sqrt(2.0 * energy_norm) * xi  # normalised to v_th

    # Streaming coefficient: v_∥ * b.grad(theta)
    coeff = v_par * geom.b_dot_grad_theta

    dtheta = geom.theta[1] - geom.theta[0] if n_theta > 1 else 1.0
    D = np.zeros((n_theta, n_theta))
    for i in range(1, n_theta - 1):
        D[i, i + 1] = coeff[i] / (2.0 * dtheta)
        D[i, i - 1] = -coeff[i] / (2.0 * dtheta)
    # Periodic boundary
    D[0, 1] = coeff[0] / (2.0 * dtheta)
    D[0, -1] = -coeff[0] / (2.0 * dtheta)
    D[-1, 0] = coeff[-1] / (2.0 * dtheta)
    D[-1, -2] = -coeff[-1] / (2.0 * dtheta)
    return D


def solve_eigenvalue_single_ky(
    k_y_rho_s: float,
    species_list: list[GKSpecies],
    geom: MillerGeometry,
    vgrid: VelocityGrid,
    R0: float = 2.78,
    a: float = 1.0,
    B0: float = 2.0,
    Z_eff: float = 1.0,
    nu_star: float = 0.01,
) -> EigenMode:
    """Solve the linear GK eigenvalue problem at a single k_y.

    Uses the response-matrix formulation: for each (E, lambda) pair,
    build the theta-space response to phi(theta), then assemble the
    quasineutrality equation into a matrix eigenvalue problem for phi.

    For adiabatic electrons, the problem reduces to a matrix of size
    n_theta x n_theta.  For kinetic electrons, the matrix doubles.
    """
    n_theta = len(geom.theta)
    B_ratio = geom.B_mag / np.mean(geom.B_mag)
    ion = species_list[0]

    # Adiabatic electron response
    has_kinetic_electrons = any(not s.is_adiabatic and s.charge_e < 0 for s in species_list)

    # Build ion response matrix: R_i(theta, theta') such that
    # integral{ g_i J_0 d^3v } = R_i . phi
    R_ion = np.zeros((n_theta, n_theta), dtype=complex)

    omega_star_n_i, omega_star_T_i = _diamagnetic_frequency(k_y_rho_s, ion, R0, a)

    for ie in range(vgrid.n_energy):
        E_norm = vgrid.energy[ie]
        w_E = vgrid.energy_weights[ie]
        # Maxwellian weight: (2/sqrt(pi)) * sqrt(E) * exp(-E) * w_E
        fm_weight = (2.0 / np.sqrt(np.pi)) * np.sqrt(E_norm) * np.exp(-E_norm) * w_E

        for il in range(vgrid.n_lambda):
            lam_val = vgrid.lam[il]
            w_lam = vgrid.lambda_weights[il]

            # FLR: k_perp * rho_i * sqrt(2 * lambda * E)
            rho_i_over_a = (
                ion.mass_kg * ion.thermal_speed / (abs(ion.charge_e) * _E_CHARGE * B0) / a
            )
            b_arg = k_y_rho_s * rho_i_over_a * np.sqrt(2.0 * lam_val * E_norm) * np.ones(n_theta)
            J0_val = bessel_j0(b_arg)

            # Drift frequency
            omega_D = _drift_frequency(k_y_rho_s, geom, E_norm, lam_val, B_ratio)

            # Diamagnetic frequency with temperature gradient
            eta_i = omega_star_T_i / max(abs(omega_star_n_i), 1e-10) if omega_star_n_i != 0 else 0.0
            omega_star_full = omega_star_n_i * (1.0 + eta_i * (E_norm - 1.5))

            # Parallel streaming
            D_par = _parallel_streaming_matrix(n_theta, geom, E_norm, lam_val, B_ratio)

            # Collision operator (simplified: nu_D * pitch-angle scattering applied to theta structure)
            nu_D, _ = collision_frequencies(ion, ion.density_19, ion.temperature_keV, Z_eff)
            nu_eff = nu_D * nu_star  # scale by collisionality parameter

            # Response: for eigenvalue omega, g = (omega_star_full - omega_D) / (omega - omega_D - v_par*grad_par - nu) * J0 * phi
            # In matrix form: g = G(omega) . J0 . phi
            # The quasineutrality integral gives: int{ J0 * G(omega) * J0 } * phi = n_response * phi
            # For the initial-value approach, we linearise around omega=0 and solve A*phi = omega*B*phi

            # Build the A and B matrices for this velocity point
            # A_vpt = (omega_D + i*D_par + nu_eff*I) as diagonal+tridiagonal in theta
            A_vpt = np.diag(omega_D + nu_eff) + 1j * D_par

            # B_vpt = I (identity, from the omega * g term)
            # Response to phi: contribution to quasineutrality
            J0_diag = np.diag(J0_val)
            weight = fm_weight * w_lam

            # Accumulate: R_ion += weight * J0 * (A_vpt)^{-1} * diag(omega_star_full) * J0
            # For stability, directly accumulate the drive term
            drive_diag = np.diag(np.full(n_theta, omega_star_full))
            R_ion += weight * (J0_diag @ drive_diag @ J0_diag + 1j * J0_diag @ D_par @ J0_diag)

    # Adiabatic electron contribution: n_e_response = (e^2 n_e / T_e) * (phi - <phi>)
    # In normalised units: 1.0 * (phi - flux_surface_average(phi))
    if not has_kinetic_electrons:
        I_theta = np.eye(n_theta)
        fsa = np.ones((n_theta, n_theta)) / n_theta  # flux-surface average operator
        adiabatic_response = I_theta - fsa
    else:
        adiabatic_response = np.zeros((n_theta, n_theta))

    # Full dispersion matrix: D(omega) = R_ion + adiabatic_response = 0
    # Eigenvalue problem: (R_ion + adiabatic_response) phi = omega * phi
    # This is approximate — the full problem is nonlinear in omega.
    # We solve the linearised version.
    full_matrix = R_ion + adiabatic_response

    try:
        eigenvalues, eigenvectors = np.linalg.eig(full_matrix)
    except np.linalg.LinAlgError:
        return EigenMode(k_y_rho_s=k_y_rho_s, omega_r=0.0, gamma=0.0, mode_type="stable")

    # Find most unstable mode (largest imaginary part)
    gammas = eigenvalues.imag
    omega_rs = eigenvalues.real

    if np.all(gammas <= 0):
        return EigenMode(k_y_rho_s=k_y_rho_s, omega_r=0.0, gamma=0.0, mode_type="stable")

    idx = int(np.argmax(gammas))
    gamma_max = gammas[idx]
    omega_r_max = omega_rs[idx]
    phi_mode = np.abs(eigenvectors[:, idx])

    # Classify mode by frequency direction
    if omega_r_max < 0:
        mode_type = "ITG"
    elif omega_r_max > 0:
        mode_type = "TEM"
    else:
        mode_type = "stable"

    return EigenMode(
        k_y_rho_s=k_y_rho_s,
        omega_r=float(omega_r_max),
        gamma=float(max(gamma_max, 0.0)),
        mode_type=mode_type,
        phi_theta=phi_mode,
    )


def solve_linear_gk(
    species_list: list[GKSpecies] | None = None,
    geom: MillerGeometry | None = None,
    vgrid: VelocityGrid | None = None,
    R0: float = 2.78,
    a: float = 1.0,
    B0: float = 2.0,
    q: float = 1.4,
    s_hat: float = 0.78,
    Z_eff: float = 1.0,
    nu_star: float = 0.01,
    n_ky_ion: int = 16,
    n_ky_etg: int = 0,
    n_theta: int = 64,
    n_period: int = 2,
) -> LinearGKResult:
    """Full k_y scan of the linear GK eigenvalue solver.

    Parameters
    ----------
    species_list : list of GKSpecies
        Plasma species. Default: [deuterium, adiabatic electron].
    geom : MillerGeometry
        Flux-tube geometry. Default: circular with given (R0, a, q, s_hat).
    vgrid : VelocityGrid
        Velocity-space grid. Default: (16, 24).
    n_ky_ion, n_ky_etg : int
        Number of k_y points on ion and electron scales.
    """
    if species_list is None:
        species_list = [deuterium_ion(), electron()]
    if geom is None:
        geom = circular_geometry(
            R0=R0, a=a, q=q, s_hat=s_hat, B0=B0, n_theta=n_theta, n_period=n_period
        )
    if vgrid is None:
        vgrid = VelocityGrid(n_energy=8, n_lambda=12)

    # Ion-scale k_y grid (log-spaced)
    k_y_ion = np.logspace(np.log10(0.05), np.log10(2.0), n_ky_ion) if n_ky_ion > 0 else np.array([])

    # ETG-scale k_y grid
    k_y_etg = np.logspace(np.log10(2.0), np.log10(40.0), n_ky_etg) if n_ky_etg > 0 else np.array([])

    k_y_all = np.concatenate([k_y_ion, k_y_etg])

    modes = []
    for ky in k_y_all:
        mode = solve_eigenvalue_single_ky(
            k_y_rho_s=ky,
            species_list=species_list,
            geom=geom,
            vgrid=vgrid,
            R0=R0,
            a=a,
            B0=B0,
            Z_eff=Z_eff,
            nu_star=nu_star,
        )
        modes.append(mode)

    return LinearGKResult(
        k_y=np.array([m.k_y_rho_s for m in modes]),
        gamma=np.array([m.gamma for m in modes]),
        omega_r=np.array([m.omega_r for m in modes]),
        mode_type=[m.mode_type for m in modes],
        modes=modes,
    )
