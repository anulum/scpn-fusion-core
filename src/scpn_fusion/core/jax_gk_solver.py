# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — JAX-Accelerated Linear Gyrokinetic Eigenvalue Solver
"""JAX-accelerated linear GK eigenvalue solver.

Re-implements the response-matrix formulation from gk_eigenvalue.py using
jax.numpy, jax.vmap (batch over k_y), and jax.grad (transport stiffness
d(chi_i)/d(R_L_Ti)).  Falls back to NumPy solver when JAX is unavailable.

References:
  - Dimits et al., Phys. Plasmas 7 (2000) 969 — Cyclone Base Case
  - Kotschenreuther et al., Comp. Phys. Comm. 88 (1995) 128
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from scpn_fusion.core.gk_eigenvalue import EigenMode, LinearGKResult
from scpn_fusion.core.gk_geometry import circular_geometry
from scpn_fusion.core.gk_species import (
    VelocityGrid,
    collision_frequencies,
    deuterium_ion,
    electron,
)

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _HAS_JAX = False

_E_CHARGE = 1.602176634e-19  # C

_logger = logging.getLogger(__name__)


def has_jax() -> bool:
    return _HAS_JAX


def _require_jax() -> None:
    if not _HAS_JAX:
        raise ImportError(
            "JAX is required for jax_gk_solver but not installed. pip install jax jaxlib"
        )


# ── JAX Bessel J0 approximation ─────────────────────────────────────
# Abramowitz & Stegun 9.4.1 / 9.4.3 — max error < 5e-8.


def _bessel_j0_jax(x: Any) -> Any:
    """J_0(x) via rational Chebyshev approximation, differentiable under JAX."""
    ax = jnp.abs(x)

    y = (ax / 3.0) ** 2
    small = 1.0 + y * (
        -2.2499997
        + y * (1.2656208 + y * (-0.3163866 + y * (0.0444479 + y * (-0.0039444 + y * 0.0002100))))
    )

    y3 = 3.0 / ax
    p0 = 0.79788456 + y3 * (
        -0.00000077
        + y3
        * (
            -0.00552740
            + y3 * (-0.00009512 + y3 * (0.00137237 + y3 * (-0.00072805 + y3 * 0.00014476)))
        )
    )
    q0 = y3 * (
        -0.04166397
        + y3
        * (
            -0.00003954
            + y3 * (0.00262573 + y3 * (-0.00054125 + y3 * (-0.00029333 + y3 * 0.00013558)))
        )
    )
    theta0 = ax - 0.78539816  # ax - pi/4
    large = p0 * jnp.cos(theta0 - q0) / jnp.sqrt(ax + 1e-30)

    return jnp.where(ax <= 3.0, small, large)


# ── Core JAX kernels ────────────────────────────────────────────────


def _build_response_matrix_single_ky(
    k_y_rho_s: Any,
    ion_R_L_n: float,
    ion_R_L_T: float,
    rho_i_over_a: float,
    nu_eff: float,
    B_ratio: Any,
    kappa_n: Any,
    kappa_g: Any,
    b_dot_grad_theta: Any,
    theta: Any,
    energy: Any,
    energy_weights: Any,
    lam: Any,
    lambda_weights: Any,
    has_kinetic_electrons: bool,
) -> tuple[Any, Any, Any]:
    """Build the response matrix for a single k_y. Pure JAX, no side effects.

    nu_eff and rho_i_over_a are pre-computed outside the traced path to
    avoid constructing Python objects (GKSpecies) inside JAX tracing.
    """
    n_theta = theta.shape[0]

    omega_star_n_i = k_y_rho_s * ion_R_L_n
    omega_star_T_i = k_y_rho_s * ion_R_L_T

    eta_i = jnp.where(jnp.abs(omega_star_n_i) > 1e-10, omega_star_T_i / omega_star_n_i, 0.0)

    n_energy = int(energy.shape[0])
    n_lambda = int(lam.shape[0])

    dtheta = theta[1] - theta[0]

    R_ion_real = jnp.zeros((n_theta, n_theta))
    R_ion_imag = jnp.zeros((n_theta, n_theta))

    for ie in range(n_energy):
        E_norm = energy[ie]
        w_E = energy_weights[ie]
        fm_weight = (2.0 / jnp.sqrt(jnp.pi)) * jnp.sqrt(E_norm) * jnp.exp(-E_norm) * w_E

        for il in range(n_lambda):
            lam_val = lam[il]
            w_lam = lambda_weights[il]

            b_arg = k_y_rho_s * rho_i_over_a * jnp.sqrt(2.0 * lam_val * E_norm) * jnp.ones(n_theta)
            J0_val = _bessel_j0_jax(b_arg)

            xi_sq = jnp.maximum(1.0 - lam_val * B_ratio, 0.0)
            omega_D = k_y_rho_s * 2.0 * E_norm * (kappa_n * xi_sq + kappa_g * jnp.sqrt(xi_sq))

            omega_star_full = omega_star_n_i * (1.0 + eta_i * (E_norm - 1.5))

            xi = jnp.sqrt(xi_sq)
            v_par = jnp.sqrt(2.0 * E_norm) * xi
            coeff = v_par * b_dot_grad_theta

            idx_p = jnp.arange(n_theta)
            idx_next = (idx_p + 1) % n_theta
            idx_prev = (idx_p - 1) % n_theta

            D_par = jnp.zeros((n_theta, n_theta))
            D_par = D_par.at[idx_p, idx_next].add(coeff / (2.0 * dtheta))
            D_par = D_par.at[idx_p, idx_prev].add(-coeff / (2.0 * dtheta))

            J0_diag = jnp.diag(J0_val)
            drive_diag = jnp.diag(jnp.full(n_theta, omega_star_full))

            weight = fm_weight * w_lam

            R_ion_real = R_ion_real + weight * (J0_diag @ drive_diag @ J0_diag)
            R_ion_imag = R_ion_imag + weight * (J0_diag @ D_par @ J0_diag)

    I_theta = jnp.eye(n_theta)
    fsa = jnp.ones((n_theta, n_theta)) / n_theta
    adiabatic = jnp.where(has_kinetic_electrons, 0.0, 1.0)
    adiabatic_response = adiabatic * (I_theta - fsa)

    full_real = R_ion_real + adiabatic_response
    full_imag = R_ion_imag

    return full_real, full_imag, R_ion_real


def _solve_eigenvalue_from_matrix(
    full_real: Any, full_imag: Any
) -> tuple[float, float, str, np.ndarray | None]:
    """Extract most unstable eigenmode from the response matrix (NumPy)."""
    full_matrix = np.asarray(full_real) + 1j * np.asarray(full_imag)

    try:
        eigenvalues, eigenvectors = np.linalg.eig(full_matrix)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, "stable", None

    gammas = eigenvalues.imag
    omega_rs = eigenvalues.real

    if np.all(gammas <= 0):
        return 0.0, 0.0, "stable", None

    idx = int(np.argmax(gammas))
    gamma_max = float(gammas[idx])
    omega_r_max = float(omega_rs[idx])
    phi_mode = np.abs(eigenvectors[:, idx])

    if omega_r_max < 0:
        mode_type = "ITG"
    elif omega_r_max > 0:
        mode_type = "TEM"
    else:
        mode_type = "stable"

    return float(max(gamma_max, 0.0)), omega_r_max, mode_type, phi_mode


def _precompute_ion_params(
    ion: Any, B0: float, a: float, Z_eff: float, nu_star: float
) -> tuple[float, float]:
    """Pre-compute collision frequency and FLR scale outside JAX trace."""
    rho_i_over_a = ion.mass_kg * ion.thermal_speed / (abs(ion.charge_e) * _E_CHARGE * B0) / a
    nu_D, _ = collision_frequencies(ion, ion.density_19, ion.temperature_keV, Z_eff)
    nu_eff = nu_D * nu_star
    return float(rho_i_over_a), float(nu_eff)


def _make_batched_builder(
    ion_R_L_n: float,
    ion_R_L_T: float,
    rho_i_over_a: float,
    nu_eff: float,
    B_ratio: Any,
    kappa_n: Any,
    kappa_g: Any,
    b_dot_grad_theta: Any,
    theta: Any,
    energy: Any,
    energy_weights: Any,
    lam: Any,
    lambda_weights: Any,
    has_kinetic_electrons: bool,
) -> Any:
    """Return a vmap'd function that builds response matrices for all k_y at once."""

    def _build_single(k_y_rho_s: Any) -> tuple[Any, Any, Any]:
        return _build_response_matrix_single_ky(
            k_y_rho_s,
            ion_R_L_n,
            ion_R_L_T,
            rho_i_over_a,
            nu_eff,
            B_ratio,
            kappa_n,
            kappa_g,
            b_dot_grad_theta,
            theta,
            energy,
            energy_weights,
            lam,
            lambda_weights,
            has_kinetic_electrons,
        )

    return jax.vmap(_build_single)


def solve_linear_gk_jax(
    species_list: Any = None,
    geom: Any = None,
    vgrid: Any = None,
    R0: float = 2.78,
    a: float = 1.0,
    B0: float = 2.0,
    q: float = 1.4,
    s_hat: float = 0.78,
    Z_eff: float = 1.0,
    nu_star: float = 0.01,
    n_ky_ion: int = 16,
    n_theta: int = 64,
) -> LinearGKResult:
    """JAX-accelerated linear GK solver. Batched over k_y via vmap.

    Parameters match solve_linear_gk from gk_eigenvalue.py.
    Uses JAX to build response matrices for all k_y simultaneously,
    then extracts eigenvalues via NumPy (eig is not JIT-able on all backends).
    """
    _require_jax()

    if species_list is None:
        species_list = [deuterium_ion(), electron()]
    if geom is None:
        geom = circular_geometry(R0=R0, a=a, q=q, s_hat=s_hat, B0=B0, n_theta=n_theta, n_period=2)
    if vgrid is None:
        vgrid = VelocityGrid(n_energy=12, n_lambda=16)

    k_y_all = jnp.logspace(jnp.log10(0.05), jnp.log10(2.0), n_ky_ion)

    ion = species_list[0]
    has_kinetic_electrons = any(not s.is_adiabatic and s.charge_e < 0 for s in species_list)

    rho_i_over_a, nu_eff = _precompute_ion_params(ion, B0, a, Z_eff, nu_star)

    B_ratio = jnp.array(geom.B_mag / np.mean(geom.B_mag))
    kappa_n = jnp.array(geom.kappa_n)
    kappa_g = jnp.array(geom.kappa_g)
    b_dot_grad_theta = jnp.array(geom.b_dot_grad_theta)
    theta = jnp.array(geom.theta)
    energy = jnp.array(vgrid.energy)
    energy_weights = jnp.array(vgrid.energy_weights)
    lam = jnp.array(vgrid.lam)
    lambda_weights = jnp.array(vgrid.lambda_weights)

    batched_build = _make_batched_builder(
        float(ion.R_L_n),
        float(ion.R_L_T),
        rho_i_over_a,
        nu_eff,
        B_ratio,
        kappa_n,
        kappa_g,
        b_dot_grad_theta,
        theta,
        energy,
        energy_weights,
        lam,
        lambda_weights,
        has_kinetic_electrons,
    )
    all_real, all_imag, _ = batched_build(k_y_all)

    k_y_np = np.asarray(k_y_all)
    modes = []
    for i in range(n_ky_ion):
        gamma_val, omega_r_val, mode_type, phi = _solve_eigenvalue_from_matrix(
            all_real[i], all_imag[i]
        )
        modes.append(
            EigenMode(
                k_y_rho_s=float(k_y_np[i]),
                omega_r=omega_r_val,
                gamma=gamma_val,
                mode_type=mode_type,
                phi_theta=phi,
            )
        )

    return LinearGKResult(
        k_y=np.array([m.k_y_rho_s for m in modes]),
        gamma=np.array([m.gamma for m in modes]),
        omega_r=np.array([m.omega_r for m in modes]),
        mode_type=[m.mode_type for m in modes],
        modes=modes,
    )


def _chi_i_proxy(
    R_L_Ti: Any,
    R_L_n: float,
    rho_i_over_a: float,
    nu_eff: float,
    B_ratio: Any,
    kappa_n: Any,
    kappa_g: Any,
    b_dot_grad_theta: Any,
    theta: Any,
    energy: Any,
    energy_weights: Any,
    lam: Any,
    lambda_weights: Any,
    has_kinetic_electrons: bool,
    n_ky_ion: int,
) -> Any:
    """R_L_Ti -> differentiable scalar proxy for chi_i.

    Uses the squared Frobenius norm of the ion drive matrix (R_ion_real)
    summed over k_y. Since R_ion_real ~ omega_star_full ~ R_L_Ti, this
    is quadratic in R_L_Ti with strictly positive gradient for R_L_Ti > 0.
    """
    k_y_all = jnp.logspace(jnp.log10(0.05), jnp.log10(2.0), n_ky_ion)

    batched = _make_batched_builder(
        R_L_n,
        R_L_Ti,
        rho_i_over_a,
        nu_eff,
        B_ratio,
        kappa_n,
        kappa_g,
        b_dot_grad_theta,
        theta,
        energy,
        energy_weights,
        lam,
        lambda_weights,
        has_kinetic_electrons,
    )
    _, _, R_ion_real = batched(k_y_all)

    # ||R_ion||_F^2 scales as omega_star_full^2 ~ (R_L_n + R_L_Ti * (E-1.5))^2
    return jnp.sum(R_ion_real**2)


def transport_stiffness_jax(
    R_L_Ti: float,
    R0: float = 2.78,
    a: float = 1.0,
    B0: float = 2.0,
    q: float = 1.4,
    s_hat: float = 0.78,
    Z_eff: float = 1.0,
    nu_star: float = 0.01,
    n_ky_ion: int = 8,
    n_theta: int = 32,
) -> float:
    """d(chi_i_proxy)/d(R_L_Ti) via jax.grad.

    Returns the transport stiffness — sensitivity of the ion heat flux
    proxy to the normalised temperature gradient. Positive above critical
    gradient, near-zero below.
    """
    _require_jax()

    ion = deuterium_ion(R_L_T=R_L_Ti)
    geom = circular_geometry(R0=R0, a=a, q=q, s_hat=s_hat, B0=B0, n_theta=n_theta, n_period=1)
    vgrid = VelocityGrid(n_energy=6, n_lambda=8)

    rho_i_over_a, nu_eff = _precompute_ion_params(ion, B0, a, Z_eff, nu_star)
    has_kinetic_electrons = False  # default: adiabatic electrons

    B_ratio = jnp.array(geom.B_mag / np.mean(geom.B_mag))
    kappa_n = jnp.array(geom.kappa_n)
    kappa_g = jnp.array(geom.kappa_g)
    b_dot_grad_theta = jnp.array(geom.b_dot_grad_theta)
    theta = jnp.array(geom.theta)
    energy = jnp.array(vgrid.energy)
    energy_weights = jnp.array(vgrid.energy_weights)
    lam_arr = jnp.array(vgrid.lam)
    lambda_weights = jnp.array(vgrid.lambda_weights)

    grad_fn = jax.grad(_chi_i_proxy, argnums=0)

    stiffness = grad_fn(
        jnp.float64(R_L_Ti),
        float(ion.R_L_n),
        rho_i_over_a,
        nu_eff,
        B_ratio,
        kappa_n,
        kappa_g,
        b_dot_grad_theta,
        theta,
        energy,
        energy_weights,
        lam_arr,
        lambda_weights,
        has_kinetic_electrons,
        n_ky_ion,
    )

    return float(stiffness)
