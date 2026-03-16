# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Quasilinear Gyrokinetic Transport Model
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Physical constants
# m_p = 1.67262192369e-27  kg


@dataclass
class GyrokineticsParams:
    """TGLF-10 style input vector for the quasilinear model."""

    R_L_Ti: float
    R_L_Te: float
    R_L_ne: float
    q: float
    s_hat: float
    alpha_MHD: float
    Te_Ti: float
    Z_eff: float
    nu_star: float
    beta_e: float
    epsilon: float = 0.1  # r / R


@dataclass
class SpectrumResult:
    """Computed instability spectrum."""

    k_y: np.ndarray
    gamma_linear: np.ndarray
    omega_r: np.ndarray
    mode_type: np.ndarray  # 0: stable, 1: ITG, 2: TEM, 3: ETG


@dataclass
class TransportFluxes:
    """Quasilinear fluxes and effective diffusivities."""

    chi_i: float
    chi_e: float
    D_e: float


def solve_dispersion(
    params: GyrokineticsParams, k_theta_rho_s: float, etg_scale: bool = False
) -> tuple[float, float, int]:
    """
    Solve the local electrostatic dispersion relation for growth rate gamma and real frequency omega_r.

    Parameters
    ----------
    params : GyrokineticsParams
        Local plasma parameters.
    k_theta_rho_s : float
        Normalized perpendicular wavenumber.
    etg_scale : bool
        If True, evaluate the ETG mode dispersion.

    Returns
    -------
    gamma : float
        Growth rate [c_s / R]
    omega_r : float
        Real frequency [c_s / R]
    mode_type : int
        1 for ITG, 2 for TEM, 3 for ETG, 0 for stable
    """
    k_y = k_theta_rho_s

    if etg_scale:
        # ETG mode (Jenko et al. 2000)
        # R/L_Te_crit = (1 + Z_eff) * max(1.33 + 1.91 s_hat/q, 0)
        R_L_Te_crit = (1.0 + params.Z_eff) * max(1.33 + 1.91 * params.s_hat / params.q, 0.0)
        drive = params.R_L_Te - R_L_Te_crit
        if drive > 0.0:
            gamma = k_y * params.R_L_Te * np.sqrt(drive) / (1.0 + k_y**2)
            omega_r = k_y * params.R_L_Te  # omega_*Te
            return gamma, omega_r, 3
        return 0.0, 0.0, 0

    # ITG mode
    gamma_ITG = 0.0
    omega_ITG = 0.0
    # Dimits shift included
    R_L_Ti_crit = max(
        (4.0 / 3.0) * (1.0 + 1.0 / params.Te_Ti) * (1.0 + 2.0 * params.s_hat / params.q),
        0.0,
    )
    drive_ITG = params.R_L_Ti - R_L_Ti_crit
    if drive_ITG > 0.0:
        gamma_ITG = k_y * params.R_L_Ti * np.sqrt(drive_ITG) / (1.0 + k_y**2)
        # Ion diamagnetic direction
        omega_ITG = -k_y * params.R_L_Ti / params.Te_Ti

    # TEM mode
    gamma_TEM = 0.0
    omega_TEM = 0.0
    f_t = np.sqrt(2.0 * params.epsilon / (1.0 + params.epsilon))
    # Approximate omega_be_norm from nu_star. nu_star = nu_eff / omega_be
    # Thus nu_eff / omega_be = nu_star
    nu_eff_over_omega_be = params.nu_star
    omega_star_e = k_y * params.R_L_ne

    # Romanelli & Zonca 1993 TEM model
    drive_TEM = omega_star_e
    if drive_TEM > 0.0:
        gamma_TEM = f_t * omega_star_e / (1.0 + k_y**2 * (1.0 + nu_eff_over_omega_be))
        omega_TEM = omega_star_e

    # Identify dominant mode
    if gamma_ITG > gamma_TEM and gamma_ITG > 0.0:
        return gamma_ITG, omega_ITG, 1
    elif gamma_TEM > gamma_ITG and gamma_TEM > 0.0:
        return gamma_TEM, omega_TEM, 2

    return 0.0, 0.0, 0


def compute_spectrum(
    params: GyrokineticsParams, n_modes: int = 16, include_etg: bool = False
) -> SpectrumResult:
    """
    Scan k_theta rho_s and compute growth rate spectrum.
    """
    k_y_list = []
    gamma_list = []
    omega_list = []
    type_list = []

    # Ion scale (ITG/TEM)
    k_y_ion = np.linspace(0.1, 2.0, n_modes)
    for ky in k_y_ion:
        g, w, mt = solve_dispersion(params, ky, etg_scale=False)
        k_y_list.append(ky)
        gamma_list.append(g)
        omega_list.append(w)
        type_list.append(mt)

    # Electron scale (ETG)
    if include_etg:
        k_y_elec = np.linspace(2.0, 30.0, n_modes)
        for ky in k_y_elec:
            g, w, mt = solve_dispersion(params, ky, etg_scale=True)
            # Normalization factor for ETG (c_e / R vs c_s / R)
            # Actually, standard TGLF treats them together but with sqrt(m_i/m_e)
            # Let's scale ETG gamma by sqrt(m_i/m_e) ~ 60 to put it in c_s/R units
            # For deuterium, sqrt(m_i/m_e) = 60.6
            mass_ratio_sqrt = 60.6
            k_y_list.append(ky)
            gamma_list.append(g * mass_ratio_sqrt)
            omega_list.append(w * mass_ratio_sqrt)
            type_list.append(mt)

    return SpectrumResult(
        np.array(k_y_list),
        np.array(gamma_list),
        np.array(omega_list),
        np.array(type_list, dtype=int),
    )


def quasilinear_fluxes(params: GyrokineticsParams, spectrum: SpectrumResult) -> TransportFluxes:
    """
    Apply saturation rule and return effective diffusivities.
    """
    # gamma_max = c_s / (q R) => normalized gamma_max = 1 / q
    gamma_max = 1.0 / max(params.q, 0.1)

    chi_i = 0.0
    chi_e = 0.0
    D_e = 0.0

    for i in range(len(spectrum.k_y)):
        ky = spectrum.k_y[i]
        gamma_lin = spectrum.gamma_linear[i]
        omega_r = spectrum.omega_r[i]
        mt = spectrum.mode_type[i]

        if gamma_lin <= 0.0 or mt == 0:
            continue

        # Saturation rule
        gamma_sat = gamma_lin / (1.0 + gamma_lin / gamma_max)

        # Mixing length amplitude
        phi_sq = 1.0 / ky**2

        # Quasilinear weights
        # Q_s = sum gamma_sat * phi_sq * (omega_*Ts / omega_r) * n_s T_s
        # chi_s = sum gamma_sat * phi_sq * (omega_*Ts / omega_r) / (R/L_Ts) * R
        # In normalized units (chi_s / chi_gB):
        # chi_s_norm = sum gamma_sat * phi_sq * (omega_*Ts / omega_r) / R_L_Ts

        if mt == 1:  # ITG
            # omega_*Ti = - ky * R_L_Ti / Te_Ti
            omega_star_Ti = -ky * params.R_L_Ti / params.Te_Ti
            weight_i = omega_star_Ti / omega_r if omega_r != 0.0 else 0.0
            if params.R_L_Ti > 0:
                chi_i += gamma_sat * phi_sq * weight_i

        elif mt == 2:  # TEM
            omega_star_Te = ky * params.R_L_Te
            omega_star_n = ky * params.R_L_ne
            weight_e = omega_star_Te / omega_r if omega_r != 0.0 else 0.0
            weight_n = omega_star_n / omega_r if omega_r != 0.0 else 0.0

            if params.R_L_Te > 0:
                chi_e += gamma_sat * phi_sq * weight_e
            if params.R_L_ne > 0:
                D_e += gamma_sat * phi_sq * weight_n

        elif mt == 3:  # ETG
            omega_star_Te = ky * params.R_L_Te
            weight_e = omega_star_Te / omega_r if omega_r != 0.0 else 0.0
            # For ETG, scale by (rho_e/rho_s)^2 ~ 1/3600
            # Actually, standard TGLF has separate scaling.
            # We already scaled gamma by 60. 1/ky^2 is in 1/(k_y_e)^2 = (rho_e/rho_s)^2 / (k_theta rho_s)^2
            # So multiply by 1 / 60.6**2
            mass_ratio_sq = 60.6**2
            if params.R_L_Te > 0:
                chi_e += (gamma_sat * phi_sq * weight_e) / mass_ratio_sq

    # Ensure positivity
    chi_i = max(chi_i, 0.0)
    chi_e = max(chi_e, 0.0)
    D_e = max(D_e, 0.0)

    return TransportFluxes(chi_i, chi_e, D_e)


class GyrokineticTransportModel:
    """
    Drop-in replacement for Gyro-Bohm transport scaling.
    """

    def __init__(self, n_modes: int = 16, include_etg: bool = False):
        self.n_modes = n_modes
        self.include_etg = include_etg
        # Typical tuning constant for macroscopic match
        self.c_tune = 0.5

    def evaluate(self, rho: float, profiles: dict[str, Any]) -> tuple[float, float, float]:
        """
        Evaluate transport coefficients at a single radial point.
        """
        if rho <= 0.05:
            # Axis boundary
            return 0.01, 0.01, 0.01

        # Extract local gradients and parameters
        R0 = profiles.get("R0", 2.0)
        a = profiles.get("a", 0.5)
        B0 = profiles.get("B0", 1.0)
        q = profiles.get("q", 1.0)
        s_hat = profiles.get("s_hat", 1.0)
        Te = profiles.get("Te", 1.0)
        Ti = profiles.get("Ti", 1.0)
        ne = profiles.get("ne", 1.0)
        Z_eff = profiles.get("Z_eff", 1.5)
        dTe_dr = profiles.get("dTe_dr", 0.0)
        dTi_dr = profiles.get("dTi_dr", 0.0)
        dne_dr = profiles.get("dne_dr", 0.0)

        # Gradients R/L
        # L_x = - x / (dx/dr) => R/L_x = - R/x * dx/dr
        R_L_Te = -R0 / max(Te, 1e-3) * dTe_dr
        R_L_Ti = -R0 / max(Ti, 1e-3) * dTi_dr
        R_L_ne = -R0 / max(ne, 1e-3) * dne_dr

        # Clamp gradients to reasonable bounds for stability
        R_L_Te = max(0.0, R_L_Te)
        R_L_Ti = max(0.0, R_L_Ti)
        R_L_ne = max(0.0, R_L_ne)

        # Secondary parameters
        Te_Ti = max(Te / max(Ti, 1e-3), 0.1)
        epsilon = max(rho * a / R0, 1e-3)

        # Collisionality estimate
        # nu_star ~ R * q / (v_te * tau_e * eps^1.5)
        # We can just use a proxy or 0.1 if not fully provided
        nu_star = profiles.get("nu_star", 0.1)
        beta_e = profiles.get("beta_e", 0.01)
        alpha_MHD = profiles.get("alpha_MHD", 0.0)

        params = GyrokineticsParams(
            R_L_Ti=R_L_Ti,
            R_L_Te=R_L_Te,
            R_L_ne=R_L_ne,
            q=max(q, 0.5),
            s_hat=s_hat,
            alpha_MHD=alpha_MHD,
            Te_Ti=Te_Ti,
            Z_eff=Z_eff,
            nu_star=nu_star,
            beta_e=beta_e,
            epsilon=epsilon,
        )

        spec = compute_spectrum(params, self.n_modes, self.include_etg)
        fluxes = quasilinear_fluxes(params, spec)

        # Convert normalized chi to physical units (Gyro-Bohm scaling)
        # chi_gB = rho_s^2 * c_s / a (or R)
        # c_s = sqrt(Te / m_i)
        m_i = 2.0 * 1.6726219e-27
        e_charge = 1.602176634e-19
        Te_J = Te * 1e3 * e_charge
        c_s = np.sqrt(Te_J / m_i)
        rho_s = m_i * c_s / (e_charge * B0)
        chi_gB = rho_s**2 * c_s / R0

        chi_i_phys = fluxes.chi_i * chi_gB * self.c_tune
        chi_e_phys = fluxes.chi_e * chi_gB * self.c_tune
        D_e_phys = fluxes.D_e * chi_gB * self.c_tune

        return chi_i_phys, chi_e_phys, D_e_phys

    def evaluate_profile(
        self, rho: np.ndarray, profiles: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate full radial profile.
        """
        nr = len(rho)
        chi_i = np.zeros(nr)
        chi_e = np.zeros(nr)
        D_e = np.zeros(nr)

        R0 = profiles.get("R0", 2.0)
        a = profiles.get("a", 0.5)
        B0 = profiles.get("B0", 1.0)

        for i in range(nr):
            if rho[i] <= 0.05:
                chi_i[i] = 0.01
                chi_e[i] = 0.01
                D_e[i] = 0.01
                continue

            local_profs = {
                "R0": R0,
                "a": a,
                "B0": B0,
                "q": profiles["q"][i] if "q" in profiles else 1.0,
                "s_hat": profiles["s_hat"][i] if "s_hat" in profiles else 1.0,
                "Te": profiles["Te"][i] if "Te" in profiles else 1.0,
                "Ti": profiles["Ti"][i] if "Ti" in profiles else 1.0,
                "ne": profiles["ne"][i] if "ne" in profiles else 1.0,
                "dTe_dr": profiles["dTe_dr"][i] if "dTe_dr" in profiles else 0.0,
                "dTi_dr": profiles["dTi_dr"][i] if "dTi_dr" in profiles else 0.0,
                "dne_dr": profiles["dne_dr"][i] if "dne_dr" in profiles else 0.0,
                "nu_star": profiles["nu_star"][i] if "nu_star" in profiles else 0.1,
                "beta_e": profiles["beta_e"][i] if "beta_e" in profiles else 0.01,
                "alpha_MHD": profiles["alpha_MHD"][i] if "alpha_MHD" in profiles else 0.0,
                "Z_eff": profiles["Z_eff"][i] if "Z_eff" in profiles else 1.5,
            }
            ci, ce, de = self.evaluate(rho[i], local_profs)
            chi_i[i] = ci
            chi_e[i] = ce
            D_e[i] = de

        return chi_i, chi_e, D_e
