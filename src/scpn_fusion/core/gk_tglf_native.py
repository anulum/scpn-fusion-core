# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Native TGLF-Equivalent Transport Model
"""
Native TGLF-equivalent quasilinear transport model with SAT0/SAT1/SAT2
spectral saturation, E×B shear quench, multi-scale ITG-ETG coupling,
and trapped-particle damping.  No external binary.

References:
  - Staebler et al., Phys. Plasmas 14 (2007) 055909 — SAT0/SAT1
  - Staebler et al., Phys. Plasmas 24 (2017) 055906 — SAT2
  - Maeyama et al., Phys. Rev. Lett. 114 (2015) 255002 — cross-scale
  - Waltz et al., Phys. Plasmas 4 (1997) 2482 — E×B shear quench
  - Connor et al., Nucl. Fusion 14 (1974) 185 — trapped-particle modes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.gk_eigenvalue import LinearGKResult, solve_linear_gk
from scpn_fusion.core.gk_interface import GKLocalParams, GKOutput, GKSolverBase
from scpn_fusion.core.gk_quasilinear import mixing_length_saturation
from scpn_fusion.core.gk_species import deuterium_ion, electron

_E_CHARGE = 1.602176634e-19
_M_PROTON = 1.67262192369e-27

# Staebler et al. 2007, Table I
ALPHA_EXB_DEFAULT = 0.67
# Staebler et al. 2017, Eq. (12); Maeyama et al. 2015, Fig. 3
ALPHA_CS_DEFAULT = 3.0
# Ion/ETG scale boundary
KY_ETG_BOUNDARY = 2.0


@dataclass
class TGLFNativeConfig:
    """SAT model selection and solver parameters."""

    sat_model: str = "SAT1"
    exb_shear_model: str = "linear"
    multiscale: bool = False
    n_ky_ion: int = 16
    n_ky_etg: int = 0
    n_theta: int = 32
    alpha_exb: float = ALPHA_EXB_DEFAULT
    alpha_cs: float = ALPHA_CS_DEFAULT


@dataclass
class TGLFNativeResult:
    """Transport fluxes and spectral diagnostics."""

    chi_i: float
    chi_e: float
    D_e: float
    D_i: float = 0.0
    V_e: float = 0.0

    k_y: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    gamma: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    gamma_net: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    phi_sq: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))

    gamma_exb: float = 0.0
    dominant_mode: str = "stable"
    sat_model: str = "SAT0"
    converged: bool = True
    chi_e_etg: float = 0.0


# ---------------------------------------------------------------------------
# Helper physics functions
# ---------------------------------------------------------------------------


def exb_shear_rate(params: GKLocalParams) -> float:
    """E×B shearing rate normalised to c_s/a.

    Proxy: γ_ExB ≈ |s_hat/q| × ε × R/L_Ti × 0.1
    Waltz et al. 1997, Eq. (3).
    """
    if params.q < 1e-10:
        return 0.0
    return abs(params.s_hat / params.q) * params.epsilon * params.R_L_Ti * 0.1


def trapped_fraction(epsilon: float) -> float:
    """f_t = sqrt(2ε/(1+ε)).  Wesson, Tokamaks, Ch. 3."""
    eps = max(epsilon, 1e-6)
    return float(np.sqrt(2.0 * eps / (1.0 + eps)))


def trapped_particle_damping(params: GKLocalParams) -> float:
    """Growth-rate damping factor from trapped particles.

    Connor et al. 1974: scales with f_t × ν*/ω_bounce.
    Returns multiplicative factor in (0.1, 1.0].
    """
    f_t = trapped_fraction(params.epsilon)
    return float(max(1.0 - f_t * params.nu_star, 0.1))


def gamma_0_flr(b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Γ₀(b) = I₀(b)e^{-b} ≈ 1/(1+b).  Padé approximant, <5% error for b<3."""
    return cast(NDArray[np.float64], 1.0 / (1.0 + np.maximum(b, 0.0)))


def spectral_weight(
    gamma_net: NDArray[np.float64],
    k_y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalised spectral intensity I_k = (γ_net/k_y) / Σ(γ_net/k_y).

    Staebler 2007, Eq. (7).
    """
    raw = np.where(gamma_net > 0, gamma_net / np.maximum(k_y, 1e-10), 0.0)
    total = raw.sum()
    if total < 1e-30:
        return np.zeros_like(gamma_net)
    return np.asarray(raw / total, dtype=np.float64)


# ---------------------------------------------------------------------------
# SAT models
# ---------------------------------------------------------------------------


def sat0(
    linear: LinearGKResult,
    gamma_exb: float,
    tp_factor: float,
    cfg: TGLFNativeConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """SAT0: mixing-length saturation with E×B shear quench.

    φ²_k = γ_net / (k_y² |ω_r|), where γ_net = max(γ·tp - α|γ_ExB|, 0).
    """
    gamma_net = np.maximum(linear.gamma * tp_factor - cfg.alpha_exb * abs(gamma_exb), 0.0)
    phi_sq = mixing_length_saturation(gamma_net, linear.omega_r, linear.k_y)
    return phi_sq, gamma_net


def sat1(
    linear: LinearGKResult,
    gamma_exb: float,
    tp_factor: float,
    cfg: TGLFNativeConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """SAT1: spectral saturation with E×B shear quench.

    Staebler et al. 2007: amplitude set by peak mode, distributed by
    spectral weight.  φ²_k = I_k × γ_max_net / k_y_max².
    """
    gamma_net = np.maximum(linear.gamma * tp_factor - cfg.alpha_exb * abs(gamma_exb), 0.0)
    I_k = spectral_weight(gamma_net, linear.k_y)

    if gamma_net.max() <= 0:
        return np.zeros_like(gamma_net), gamma_net

    idx_max = int(np.argmax(gamma_net))
    peak_amp = gamma_net[idx_max] / max(linear.k_y[idx_max] ** 2, 1e-10)
    phi_sq = I_k * peak_amp
    return phi_sq, gamma_net


def sat2(
    linear: LinearGKResult,
    gamma_exb: float,
    tp_factor: float,
    cfg: TGLFNativeConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """SAT2: multi-scale spectral saturation with ITG-ETG cross-scale coupling.

    Staebler et al. 2017: ion-scale follows SAT1.  ETG modes enhanced by
    α_cs × (γ_ETG / γ_ITG_max) from Maeyama et al. 2015.
    """
    phi_sq, gamma_net = sat1(linear, gamma_exb, tp_factor, cfg)

    etg_mask = linear.k_y > KY_ETG_BOUNDARY
    ion_mask = ~etg_mask

    gamma_itg_max = (
        gamma_net[ion_mask].max() if ion_mask.any() and gamma_net[ion_mask].max() > 0 else 1e-10
    )

    if etg_mask.any():
        gamma_etg = gamma_net[etg_mask]
        where_unstable = gamma_etg > 0
        if where_unstable.any():
            cross_scale = cfg.alpha_cs * gamma_etg / gamma_itg_max
            phi_sq_etg = phi_sq[etg_mask].copy()
            phi_sq_etg[where_unstable] *= 1.0 + cross_scale[where_unstable]
            phi_sq[etg_mask] = phi_sq_etg

    return phi_sq, gamma_net


# ---------------------------------------------------------------------------
# Quasilinear flux computation
# ---------------------------------------------------------------------------


def quasilinear_weights(
    linear: LinearGKResult,
    phi_sq: NDArray[np.float64],
    gamma_net: NDArray[np.float64],
    ion_mass_amu: float,
    ion_T_keV: float,
    params: GKLocalParams,
) -> tuple[float, float, float, float, float]:
    """Velocity-space integrated QL weights → physical fluxes.

    Returns (chi_i, chi_e, D_e, V_e, chi_e_etg) in [m²/s].
    """
    m_i = ion_mass_amu * _M_PROTON
    T_i_J = ion_T_keV * 1e3 * _E_CHARGE
    c_s = np.sqrt(T_i_J / m_i)
    rho_s = m_i * c_s / (_E_CHARGE * params.B0)
    chi_gB = rho_s**2 * c_s / params.a

    # FLR: b_i = (k_y ρ_i / a)²
    rho_i = m_i * np.sqrt(2.0 * T_i_J / m_i) / (_E_CHARGE * params.B0)
    b_i = linear.k_y**2 * (rho_i / params.a) ** 2
    G0 = gamma_0_flr(b_i)

    chi_i_n = 0.0
    chi_e_n = 0.0
    chi_e_etg_n = 0.0
    D_e_n = 0.0
    V_e_n = 0.0

    for i in range(len(linear.k_y)):
        if gamma_net[i] <= 0 or phi_sq[i] <= 0:
            continue
        ky = linear.k_y[i]
        omega_r = linear.omega_r[i]
        if abs(omega_r) < 1e-10:
            continue

        amp = phi_sq[i]
        mt = linear.mode_type[i]

        if mt in ("ITG", "TEM"):
            W_i = ky * params.R_L_Ti / abs(omega_r)
            W_e = ky * params.R_L_Te / abs(omega_r)
            W_n = ky * params.R_L_ne / abs(omega_r)
            chi_i_n += amp * W_i * G0[i]
            chi_e_n += amp * W_e
            D_e_n += amp * W_n
            # Thermodiffusion pinch, Connor & Wilson 1994
            V_e_n += amp * W_n * 1.5 * params.Te_Ti
        elif mt == "ETG":
            W_e = ky * params.R_L_Te / abs(omega_r)
            # (m_e/m_i) scaling ≈ 1/3600
            val = amp * W_e / 60.0**2
            chi_e_n += val
            chi_e_etg_n += val

    return (
        float(chi_i_n * chi_gB),
        float(chi_e_n * chi_gB),
        float(D_e_n * chi_gB),
        float(V_e_n * chi_gB),
        float(chi_e_etg_n * chi_gB),
    )


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

_SAT_DISPATCH = {"SAT0": sat0, "SAT1": sat1, "SAT2": sat2}


class TGLFNativeSolver(GKSolverBase):
    """Native TGLF-equivalent quasilinear transport model.

    Pure Python implementation of SAT0/SAT1/SAT2 spectral saturation
    applied to the native linear GK eigenvalue spectrum.
    """

    def __init__(self, config: TGLFNativeConfig | None = None):
        self.config = config or TGLFNativeConfig()
        if self.config.sat_model == "SAT2":
            self.config.multiscale = True
            if self.config.n_ky_etg == 0:
                self.config.n_ky_etg = 8

    def is_available(self) -> bool:
        return True

    def prepare_input(self, params: GKLocalParams) -> Path:
        raise NotImplementedError("Native solver; use run_from_params()")

    def run(self, input_path: Path, *, timeout_s: float = 30.0) -> GKOutput:
        raise NotImplementedError("Native solver; use run_from_params()")

    def run_from_params(self, params: GKLocalParams, *, timeout_s: float = 30.0) -> GKOutput:
        r = self.solve(params)
        return GKOutput(
            chi_i=r.chi_i,
            chi_e=r.chi_e,
            D_e=r.D_e,
            D_i=r.D_i,
            gamma=r.gamma,
            omega_r=r.gamma_net,
            k_y=r.k_y,
            dominant_mode=r.dominant_mode,
            converged=r.converged,
        )

    def solve(self, params: GKLocalParams) -> TGLFNativeResult:
        cfg = self.config

        ion = deuterium_ion(
            T_keV=params.T_i_keV,
            n_19=params.n_e,
            R_L_T=params.R_L_Ti,
            R_L_n=params.R_L_ne,
        )
        elec = electron(
            T_keV=params.T_e_keV,
            n_19=params.n_e,
            R_L_T=params.R_L_Te,
            R_L_n=params.R_L_ne,
        )

        linear = solve_linear_gk(
            species_list=[ion, elec],
            R0=params.R0,
            a=params.a,
            B0=params.B0,
            q=params.q,
            s_hat=params.s_hat,
            n_ky_ion=cfg.n_ky_ion,
            n_ky_etg=cfg.n_ky_etg if cfg.multiscale else 0,
            n_theta=cfg.n_theta,
            n_period=1,
        )

        if len(linear.k_y) == 0 or linear.gamma_max <= 0:
            return TGLFNativeResult(chi_i=0.0, chi_e=0.0, D_e=0.0, sat_model=cfg.sat_model)

        gamma_exb_val = exb_shear_rate(params)
        tp_factor = trapped_particle_damping(params)

        sat_fn = _SAT_DISPATCH.get(cfg.sat_model)
        if sat_fn is None:
            raise ValueError(f"Unknown SAT model: {cfg.sat_model}")
        phi_sq, gamma_net = sat_fn(linear, gamma_exb_val, tp_factor, cfg)

        chi_i, chi_e, D_e, V_e, chi_e_etg = quasilinear_weights(
            linear, phi_sq, gamma_net, 2.0, params.T_i_keV, params
        )

        idx_max = int(np.argmax(gamma_net))
        dominant = linear.mode_type[idx_max] if gamma_net[idx_max] > 0 else "stable"

        return TGLFNativeResult(
            chi_i=chi_i,
            chi_e=chi_e,
            D_e=D_e,
            V_e=V_e,
            k_y=linear.k_y,
            gamma=linear.gamma,
            gamma_net=gamma_net,
            phi_sq=phi_sq,
            gamma_exb=gamma_exb_val,
            dominant_mode=dominant,
            sat_model=cfg.sat_model,
            converged=True,
            chi_e_etg=chi_e_etg,
        )
